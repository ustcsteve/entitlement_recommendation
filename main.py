import os
import json
import logging
from typing import List, Dict, Any, Literal, Optional
from typing_extensions import TypedDict
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field

# FastAPI Modules
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse

# LangChain, LangGraph & AWS Bedrock Modules
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrockConverse
from langgraph.graph import StateGraph, END

# Configure Enterprise Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EntitlementMicroservice")

# =====================================================================
# 1. FASTAPI REQUEST / RESPONSE SCHEMAS
# =====================================================================

class EntitlementRequest(BaseModel):
    entitlement_name: str = Field(..., description="Unique corporate identifier code.", examples=["ACH_CRED_CR_QA"])
    description: str = Field(..., description="Raw human-authored description text.", examples=["Allows Alice to draft files."])
    environment: str = Field(..., description="Target landscape. Must be: dev, qa, prod, test, uat.", examples=["qa"])
    max_revision_attempts: Optional[int] = Field(default=3, ge=1, le=5, description="Maximum loop iterations.")

class ExtractedAttributesSchema(BaseModel):
    target_users: str
    access_level: str
    resource_name: str

class EntitlementResponse(BaseModel):
    input_entitlement_name: str
    input_entitlement_desc: str
    environment: str
    loop_count: int
    extracted_attributes: Optional[ExtractedAttributesSchema] = None
    recommended_desc: str
    
    # Lineage Audit Requirements (Renamed & Added final elements)
    initial_quality_score: Optional[int] = None
    initial_quality_reason: List[str] = []
    final_quality_score: int
    final_quality_reason: List[str] = []
    
    initial_validation_passed: Optional[bool] = None
    initial_critique: List[str] = []  # Renamed from initial_final_critique
    final_validation_passed: bool
    final_critique: List[str] = []

# =====================================================================
# 2. STATE TYPE DEFINITION
# =====================================================================

class AgentCoreState(TypedDict):
    input_entitlement_name: str
    input_entitlement_desc: str
    environment: str
    loop_count: int
    max_loops: int
    extracted_attributes: Optional[Dict[str, Any]]
    recommended_desc: str
    
    quality_indicator: Literal["Yes", "No"]
    quality_score: int
    quality_reason: List[str]
    
    validation_passed: bool
    final_critique: List[str]
    current_agent: str
    
    # Audit Lineage Markers
    initial_quality_score: Optional[int]
    initial_quality_reason: List[str]
    initial_validation_passed: Optional[bool]
    initial_critique: List[str]

# =====================================================================
# 3. AWS STRUCTURAL COMPLIANCE SCHEMAS
# =====================================================================

class AccessorSchema(BaseModel):
    quality_indicator: Literal["Yes", "No"]
    quality_score: int
    quality_reasons: List[str]

class CreatorSchema(BaseModel):
    target_users: str
    access_level: str
    resource_name: str
    recommended_desc: str

class ValidatorSchema(BaseModel):
    validation_passed: bool
    guardrail_critique: List[str]

# =====================================================================
# 4. BEDROCK CHAT AGENT FACTORY (CROSS-REGION INFERENCE)
# =====================================================================

class BedrockBaseAgent:
    def __init__(self, agent_name: str, system_prompt: str, output_schema: Any, region: str = "us-east-1"):
        self.agent_name = agent_name
        self.model = ChatBedrockConverse(
            model_id="us.amazon.nova-2-lite-v1:0",
            region_name=region,
            temperature=0.0
        ).with_structured_output(output_schema, method="function_calling")
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{runtime_input}")
        ])
        self.executor = self.prompt_template | self.model

    def execute_agent(self, data: str) -> Any:
        return self.executor.invoke({"runtime_input": data})

# Initialize Agents
accessor_agent = BedrockBaseAgent(
    "AccessorAgent",
    "You are the Accessor Agent. Rate Quality Indicator 'Yes' or 'No'. Standard forms: dev, qa, prod, test, uat. Scores: 1 (missing component), 2 (unclear), 3 (compliant).",
    AccessorSchema
)
creator_agent = BedrockBaseAgent(
    "CreatorAgent",
    "You are the Creator Agent. Adhere strictly to layout blueprint structure: 'Provides [level of access] for [target users] to the [resource name] purpose [explain purpose based on context]'",
    CreatorSchema
)
validator_agent = BedrockBaseAgent(
    "ValidatorAgent",
    "You are the Validator Agent. Audit against Data Classifications: Public, Proprietary External, Internal, Confidential, Restricted. Enforce no raw human names, expand acronyms.",
    ValidatorSchema
)

# =====================================================================
# 5. LANGGRAPH RUNTIME NODES WITH AUDIT LINEAGE
# =====================================================================

def accessor_node(state: AgentCoreState) -> Dict[str, Any]:
    target_text = state.get("recommended_desc") or state["input_entitlement_desc"]
    payload = f"Entitlement: {state['input_entitlement_name']}\nEnv: {state['environment']}\nText: {target_text}"
    response: AccessorSchema = accessor_agent.execute_agent(payload)
    
    output = {
        "quality_indicator": response.quality_indicator,
        "quality_score": response.quality_score,
        "quality_reason": response.quality_reasons,
        "current_agent": "Accessor"
    }
    
    # Capture the snapshot if it's the first time the Accessor runs
    if state.get("initial_quality_score") is None:
        output["initial_quality_score"] = response.quality_score
        output["initial_quality_reason"] = response.quality_reasons
        
    return output

def creator_node(state: AgentCoreState) -> Dict[str, Any]:
    payload = (
        f"Entitlement: {state['input_entitlement_name']}\n"
        f"Context: {state['input_entitlement_desc']}\n"
        f"Prior Recommended Attempt: {state.get('recommended_desc','')}\n"
        f"Accessor Issues: {state.get('quality_reason', [])}\n"
        f"Validator Guardrail Failures: {state.get('final_critique', [])}"
    )
    response: CreatorSchema = creator_agent.execute_agent(payload)
    return {
        "extracted_attributes": {
            "target_users": response.target_users,
            "access_level": response.access_level,
            "resource_name": response.resource_name
        },
        "recommended_desc": response.recommended_desc,
        "loop_count": state.get("loop_count", 0) + 1,
        "current_agent": "Creator"
    }

def validator_node(state: AgentCoreState) -> Dict[str, Any]:
    final_desc = state.get("recommended_desc") or state["input_entitlement_desc"]
    payload = f"Name: {state['input_entitlement_name']}\nText: {final_desc}\nHistory: {state['quality_reason']}"
    response: ValidatorSchema = validator_agent.execute_agent(payload)
    
    output = {
        "validation_passed": response.validation_passed,
        "final_critique": response.guardrail_critique,
        "current_agent": "Validator"
    }
    
    # Capture the snapshot if it's the first time the Validator runs
    if state.get("initial_validation_passed") is None:
        output["initial_validation_passed"] = response.validation_passed
        output["initial_critique"] = response.guardrail_critique
        
    return output

def post_accessor_router(state: AgentCoreState) -> Literal["to_creator", "skip_to_validator"]:
    if state.get("loop_count", 0) >= state.get("max_loops", 3):
        logger.warning(f"[Loop Breaker] Max routing loops reached ({state['loop_count']}). Moving to Validator.")
        return "skip_to_validator"
    
    if state["quality_indicator"] == "No":
        return "to_creator"
    return "skip_to_validator"

def post_validator_router(state: AgentCoreState) -> Literal["loop_back_to_creator", "finalize_transaction"]:
    if state.get("loop_count", 0) >= state.get("max_loops", 3):
        return "finalize_transaction"
        
    if not state.get("validation_passed", True):
        return "loop_back_to_creator"
    return "finalize_transaction"

# --- Assemble Graph ---
builder = StateGraph(AgentCoreState)
builder.add_node("accessor", accessor_node)
builder.add_node("creator", creator_node)
builder.add_node("validator", validator_node)

builder.set_entry_point("accessor")
builder.add_conditional_edges("accessor", post_accessor_router, {"to_creator": "creator", "skip_to_validator": "validator"})
builder.add_edge("creator", "accessor")
builder.add_conditional_edges("validator", post_validator_router, {"loop_back_to_creator": "creator", "finalize_transaction": END})

agent_core_workflow = builder.compile()

# =====================================================================
# 6. FASTAPI APPLICATION DEFINITION
# =====================================================================

app = FastAPI(
    title="AgentCore Governance Multi-Agent Microservice",
    version="1.2.0",
    description="Identity governance execution API with final-state evaluation metrics tracing."
)

@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/docs")

@app.post(
    "/process-entitlement",
    response_model=EntitlementResponse,
    status_code=status.HTTP_200_OK,
    tags=["Governance Multi-Agent Engine"]
)
async def process_identity_entitlement(request: EntitlementRequest):
    try:
        logger.info(f"Processing transaction payload for asset: {request.entitlement_name}")
        
        initial_state: AgentCoreState = {
            "input_entitlement_name": request.entitlement_name,
            "input_entitlement_desc": request.description,
            "environment": request.environment.lower().strip(),
            "loop_count": 0,
            "max_loops": request.max_revision_attempts,
            "extracted_attributes": None,
            "recommended_desc": "",
            "quality_indicator": "No",
            "quality_score": 1,
            "quality_reason": [],
            "validation_passed": False,
            "final_critique": [],
            "current_agent": "HTTP_Entry_Point",
            
            # Initialize Audit Trackers
            "initial_quality_score": None,
            "initial_quality_reason": [],
            "initial_validation_passed": None,
            "initial_critique": []
        }
        
        graph_execution_result = agent_core_workflow.invoke(initial_state)
        
        # Explicit mapping mapping back cleanly to the updated EntitlementResponse schema
        response_payload = {
            "input_entitlement_name": graph_execution_result["input_entitlement_name"],
            "input_entitlement_desc": graph_execution_result["input_entitlement_desc"],
            "environment": graph_execution_result["environment"],
            "loop_count": graph_execution_result["loop_count"],
            "extracted_attributes": graph_execution_result["extracted_attributes"],
            "recommended_desc": graph_execution_result["recommended_desc"],
            
            # Historical Maps & Added Final State Snapshots
            "initial_quality_score": graph_execution_result["initial_quality_score"],
            "initial_quality_reason": graph_execution_result["initial_quality_reason"],
            "final_quality_score": graph_execution_result["quality_score"],
            "final_quality_reason": graph_execution_result["quality_reason"],  # Added
            
            "initial_validation_passed": graph_execution_result["initial_validation_passed"],
            "initial_critique": graph_execution_result["initial_critique"],  # Renamed
            "final_validation_passed": graph_execution_result["validation_passed"],
            "final_critique": graph_execution_result["final_critique"]  # Added
        }
        
        return response_payload

    except Exception as server_error:
        logger.error(f"Multi-agent loop execution failed: {str(server_error)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Identity engine execution error: {str(server_error)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)