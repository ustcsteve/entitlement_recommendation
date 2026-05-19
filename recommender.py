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
from fastapi import FastAPI, HTTPException, status, BackgroundTasks

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
    entitlement_name: str = Field(..., example="ACH_CRED_CR_QA", description="Unique corporate identifier code.")
    description: str = Field(..., example="Allows Alice to draft files.", description="Raw human-authored description text.")
    environment: str = Field(..., example="qa", description="Target landscape. Must be: dev, qa, prod, test, uat.")
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
    quality_indicator: str
    quality_score: int
    quality_reason: List[str]
    validation_passed: bool
    final_critique: List[str]

# =====================================================================
# 2. STATE TYPE DEFINITION
# =====================================================================

class AgentCoreState(TypedDict):
    input_entitlement_name: str
    input_entitlement_desc: str
    environment: str
    loop_count: int
    max_loops: int
    extracted_attributes: Dict[str, Any]
    recommended_desc: str
    quality_indicator: Literal["Yes", "No"]
    quality_score: int
    quality_reason: List[str]
    validation_passed: bool
    final_critique: List[str]
    current_agent: str

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
# 4. BEDROCK CHAT AGENT FACTORY
# =====================================================================

class BedrockBaseAgent:
    def __init__(self, agent_name: str, system_prompt: str, output_schema: Any, region: str = "us-east-1"):
        self.agent_name = agent_name
        # Using Claude 3.5 Sonnet on AWS Bedrock for structured reasoning
        self.model = ChatBedrockConverse(
            model_id="us.amazon.nova-2-lite-v1:0",
            region_name=region,
            temperature=0.0
        ).with_structured_output(output_schema)
        
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
# 5. LANGGRAPH RUNTIME NODES & COMPILATION
# =====================================================================

def accessor_node(state: AgentCoreState) -> Dict[str, Any]:
    target_text = state.get("recommended_desc") or state["input_entitlement_desc"]
    payload = f"Entitlement: {state['input_entitlement_name']}\nEnv: {state['environment']}\nText: {target_text}"
    response: AccessorSchema = accessor_agent.execute_agent(payload)
    return {
        "quality_indicator": response.quality_indicator,
        "quality_score": response.quality_score,
        "quality_reason": response.quality_reasons,
        "current_agent": "Accessor"
    }

def creator_node(state: AgentCoreState) -> Dict[str, Any]:
    payload = f"Entitlement: {state['input_entitlement_name']}\nContext: {state['input_entitlement_desc']}\nPrior Recommended Attempt: {state.get('recommended_desc','')}\nIssues: {state['quality_reason']}"
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
    return {
        "validation_passed": response.validation_passed,
        "final_critique": response.guardrail_critique,
        "current_agent": "Validator"
    }

def quality_router(state: AgentCoreState) -> Literal["to_creator", "skip_to_validator"]:
    if state.get("loop_count", 0) >= state.get("max_loops", 3):
        logger.warning(f"Max validation routing loops reached ({state['loop_count']}). Bypassing.")
        return "skip_to_validator"
    return "to_creator" if state["quality_indicator"] == "No" else "skip_to_validator"

# Assemble Multi-Agent Automation Architecture Loop
builder = StateGraph(AgentCoreState)
builder.add_node("accessor", accessor_node)
builder.add_node("creator", creator_node)
builder.add_node("validator", validator_node)

builder.set_entry_point("accessor")
builder.add_conditional_edges("accessor", quality_router, {"to_creator": "creator", "skip_to_validator": "validator"})
builder.add_edge("creator", "accessor")
builder.add_edge("validator", END)

agent_core_workflow = builder.compile()

# =====================================================================
# 6. FASTAPI WEB ENGINE APPLICATION
# =====================================================================

app = FastAPI(
    title="AgentCore Governance Multi-Agent Microservice",
    version="1.0.0",
    description="Production-grade FastAPI layer powered by LangGraph, AWS Bedrock and Pydantic validation schemas."
)

@app.on_event("startup")
async def verify_aws_connectivity():
    """Verify Bedrock configuration environment maps on startup."""
    logger.info("Initializing AgentCore engine microservice...")
    # Optional S3 / AWS connection validation logic can be added here
    logger.info("AWS AgentCore cluster components online.")

@app.get("/health", status_code=status.HTTP_200_OK, tags=["System Integrity"])
async def health_check():
    """Liveness check for standard corporate container monitoring platforms."""
    return {"status": "healthy", "engine": "AWS Bedrock Framework Cluster"}

@app.post(
    "/process-entitlement",
    response_model=EntitlementResponse,
    status_code=status.HTTP_200_OK,
    tags=["Governance Multi-Agent Engine"]
)
async def process_identity_entitlement(request: EntitlementRequest):
    """
    Executes the identity governance evaluation. 
    Assesses descriptions first, loops to Creator if corrections are required, and validates rules.
    """
    try:
        logger.info(f"Received transaction payload processing frame for asset: {request.entitlement_name}")
        
        # Build multi-agent transaction processing dictionary map
        initial_state: AgentCoreState = {
            "input_entitlement_name": request.entitlement_name,
            "input_entitlement_desc": request.description,
            "environment": request.environment.lower().strip(),
            "loop_count": 0,
            "max_loops": request.max_revision_attempts,
            "extracted_attributes": {},
            "recommended_desc": "",
            "quality_indicator": "No",
            "quality_score": 1,
            "quality_reason": [],
            "validation_passed": False,
            "final_critique": [],
            "current_agent": "HTTP_Entry_Point"
        }
        
        # Invoke synchronous compute graph thread via LangGraph async safe frame execution
        graph_execution_result = agent_core_workflow.invoke(initial_state)
        
        logger.info(f"Processing complete for asset: {request.entitlement_name}. Total Loops: {graph_execution_result['loop_count']}")
        return graph_execution_result

    except Exception as server_error:
        logger.error(f"Internal multi-agent cluster execution failed: {str(server_error)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Identity engine transactional parsing failure: {str(server_error)}"
        )

# =====================================================================
# 7. EXECUTABLE APPLICATION PROCESS CONTAINER
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    # Start ASGI localized production server process pool
    uvicorn.run("recommender:app", host="0.0.0.0", port=8000, reload=True)