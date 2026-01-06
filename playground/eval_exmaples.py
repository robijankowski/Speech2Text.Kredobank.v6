import json
from openai import OpenAI

openai_client = OpenAI(api_key="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")
openai_model = "gpt-4o"



  

def evaluate_opening_and_verification(transcript_text: str, model: str = openai_model) -> dict:
    """
    Evaluates the opening and verification section of a Kredobank collections call.
    Focuses specifically on proper introduction, identification verification, and call setup.
    
    Scoring: 0-15 points
    """
    print(f"Running evaluation for evaluate_opening_and_verification")
    SYSTEM_PROMPT = """
You are a specialist QA analyst focusing exclusively on call opening and client verification procedures for Kredobank's collections department.

Evaluate ONLY the opening and verification aspects of the call transcript according to these specific criteria:

OPENING AND VERIFICATION SCORING (0-15 points total):

**EXCELLENT (9-10 points):**
- Agent provides complete professional introduction with full name and department
- Clearly states call recording disclosure upfront
- Accurately verifies client identity using full name
- Maintains professional and courteous tone throughout opening
- Smoothly transitions to call purpose
- All regulatory requirements met flawlessly

**GOOD (7-8 points):**
- Agent provides adequate introduction with name and bank identification
- States call recording disclosure
- Verifies client identity correctly
- Professional tone maintained
- Minor gaps in completeness but core requirements met

**ACCEPTABLE (5-6 points):**
- Basic introduction provided
- Some form of recording disclosure given
- Client identity verified, though may lack precision
- Generally professional but may have minor tone issues
- Meets minimum requirements with some deficiencies

**NEEDS IMPROVEMENT (3-4 points):**
- Incomplete introduction (missing name, department, or bank identification)
- Unclear or rushed recording disclosure
- Identity verification attempted but imprecise or confusing
- Tone issues or unprofessional elements
- Several procedural gaps

**POOR (1-2 points):**
- Minimal or inadequate introduction
- Recording disclosure missing or unclear
- Identity verification problematic or skipped
- Unprofessional tone or approach
- Major procedural violations

**UNACCEPTABLE (0 points):**
- No proper introduction
- No recording disclosure
- No identity verification attempt
- Completely unprofessional opening
- Critical procedural failures

EVALUATION FOCUS AREAS:
1. **Professional Introduction**: Full name, department/bank identification, employee ID if provided
2. **Recording Disclosure**: Clear statement that call is being recorded
3. **Identity Verification**: Accurate confirmation of client's full name and identity
4. **Tone and Professionalism**: Courteous, clear, confident delivery
5. **Procedural Completeness**: All required opening elements covered
6. **Transition Quality**: Smooth movement from opening to call purpose

Pay special attention to Ukrainian banking regulations 
and Kredobank's specific protocols as evidenced in the transcript.
""".strip()

    json_schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer", 
                "minimum": 0, 
                "maximum": 15,
                "description": "Score for opening and verification (0-15 points)"
            }
        },
        "required": ["score"],
        "additionalProperties": False
    }

    try:
        response = openai_client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT                          
                },
                {
                    "role": "user",
                    "content": f"""
Evaluate ONLY the opening and verification aspects of this Kredobank collections call transcript:

Transcript:
\"\"\"
{transcript_text.strip()}
\"\"\"

Focus exclusively on how well the agent handled the call opening and client verification. Ignore all other aspects of the call such as negotiation, payment handling, or call resolution.
"""
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "opening_verification_evaluation",
                    "schema": json_schema,
                    "strict": True
                }
            }
        )

        result = json.loads(response.choices[0].message.content)
        return result["score"]  # Return just the score number

    except Exception as e:
        return {
            "error": "Opening and verification evaluation failed",
            "details": str(e),
            "raw_response": response.choices[0].message.content if 'response' in locals() else None
        }
    
def evaluate_clarity_and_accuracy(transcript_text: str, model: str = openai_model) -> int:
    """
    Evaluates the clarity and accuracy of communication in a Kredobank collections call.
    Focuses on information delivery, technical accuracy, and communication effectiveness.
    
    Scoring: 0-15 points
    """
    print(f"Running evaluation for evaluate_clarity_and_accuracy")

    SYSTEM_PROMPT = """
You are a specialist QA analyst focusing exclusively on clarity and accuracy of communication for Kredobank's collections department.

Evaluate ONLY the clarity and accuracy aspects of the call transcript according to these specific criteria:

CLARITY AND ACCURACY SCORING (0-15 points total):

**EXCELLENT (13-15 points):**
- Agent communicates all information clearly and precisely
- Financial figures, dates, and contract details are accurate and well-articulated
- Technical banking terms explained appropriately for client understanding
- Information delivered in logical, easy-to-follow sequence
- Agent demonstrates complete understanding of client's account and situation
- No confusion or misunderstandings throughout the call

**GOOD (10-12 points):**
- Most information communicated clearly with minor unclear moments
- Financial data generally accurate with good articulation
- Banking terminology mostly appropriate for client level
- Information flow is generally logical
- Agent shows good understanding of account details
- Minimal confusion or need for clarification

**ACCEPTABLE (7-9 points):**
- Basic information communicated adequately
- Some unclear explanations or minor inaccuracies
- Occasional use of confusing terminology
- Information sequence could be better organized
- Agent understanding appears adequate but not comprehensive
- Some confusion or misunderstandings that get resolved

**NEEDS IMPROVEMENT (4-6 points):**
- Several unclear or confusing explanations
- Some inaccurate information provided
- Poor use of technical terms without explanation
- Disorganized information delivery
- Agent shows incomplete understanding of situation
- Multiple instances of confusion or miscommunication

**POOR (1-3 points):**
- Frequent unclear or confusing communication
- Multiple inaccuracies in financial or account information
- Inappropriate or unexplained technical language
- Very poor information organization
- Agent demonstrates poor understanding of account
- Significant confusion throughout call

**UNACCEPTABLE (0 points):**
- Completely unclear communication
- Major inaccuracies that could mislead client
- Incomprehensible explanations
- No logical information flow
- Agent appears to have no understanding of account
- Call dominated by confusion and miscommunication

EVALUATION FOCUS AREAS:
1. **Information Accuracy**: Correct financial figures, dates, contract numbers, account details
2. **Communication Clarity**: Clear explanations, appropriate language level, easy to understand
3. **Technical Competence**: Proper use and explanation of banking terminology
4. **Logical Flow**: Information presented in coherent, logical sequence
5. **Account Knowledge**: Agent demonstrates thorough understanding of client's situation
6. **Confusion Management**: Ability to clarify misunderstandings effectively

Pay special attention to Ukrainian banking terminology and Kredobank's specific procedures as evidenced in the transcript.
""".strip()

    json_schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer", 
                "minimum": 0, 
                "maximum": 15,
                "description": "Score for clarity and accuracy (0-15 points)"
            }
        },
        "required": ["score"],
        "additionalProperties": False
    }

    try:
        response = openai_client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT                          
                },
                {
                    "role": "user",
                    "content": f"""
Evaluate ONLY the clarity and accuracy of communication in this Kredobank collections call transcript:

Transcript:
\"\"\"
{transcript_text.strip()}
\"\"\"

Focus exclusively on how clearly and accurately the agent communicated information. Ignore all other aspects of the call such as opening procedures, negotiation tactics, or call resolution.
"""
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "clarity_accuracy_evaluation",
                    "schema": json_schema,
                    "strict": True
                }
            }
        )

        result = json.loads(response.choices[0].message.content)
        return result["score"]  # Return just the score number

    except Exception as e:
        return {
            "error": "Clarity and accuracy evaluation failed",
            "details": str(e),
            "raw_response": response.choices[0].message.content if 'response' in locals() else None
        }
