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

def evaluate_addressing_questions(transcript_text: str, model: str = openai_model) -> int:
    """
    Evaluates how well the agent addresses client questions and concerns in a Kredobank collections call.
    Focuses on responsiveness, completeness, and helpfulness of answers provided.
    
    Scoring: 0-10 points
    """
    print(f"Running evaluation for evaluate_addressing_questions")

    SYSTEM_PROMPT = """
You are a specialist QA analyst focusing exclusively on how well agents address client questions and concerns for Kredobank's collections department.

Evaluate ONLY the question-answering and concern-addressing aspects of the call transcript according to these specific criteria:

ADDRESSING QUESTIONS SCORING (0-10 points total):

**EXCELLENT (9-10 points):**
- Agent responds promptly and directly to all client questions
- Provides complete, accurate, and helpful answers
- Demonstrates active listening by addressing underlying concerns
- Offers additional relevant information when appropriate
- Shows patience and willingness to clarify when needed
- Anticipates and proactively addresses potential client concerns

**GOOD (7-8 points):**
- Agent addresses most client questions adequately
- Provides generally accurate and helpful responses
- Shows good listening skills
- Offers some additional context or clarification
- Responds with appropriate patience
- Minor gaps in completeness or proactivity

**ACCEPTABLE (5-6 points):**
- Agent responds to basic client questions
- Answers are generally correct but may lack depth
- Shows adequate listening but may miss some nuances
- Provides minimal additional context
- Generally patient but may show some signs of rushing
- Addresses direct questions but doesn't anticipate concerns

**NEEDS IMPROVEMENT (3-4 points):**
- Agent addresses some questions but misses others
- Answers may be incomplete or partially inaccurate
- Poor listening evidenced by need for repetition
- Little additional context or explanation provided
- Shows impatience or rushes through responses
- Fails to address client's underlying concerns

**POOR (1-2 points):**
- Agent ignores or inadequately addresses multiple questions
- Provides inaccurate or unhelpful responses
- Poor listening skills, frequently misunderstands
- No additional context or clarification offered
- Impatient or dismissive tone when answering
- Completely misses client's main concerns

**UNACCEPTABLE (0 points):**
- Agent fails to address client questions at all
- Provides completely inaccurate or misleading information
- No evidence of listening to client concerns
- Dismissive or hostile when questioned
- Refuses to provide clarification or help
- Creates confusion rather than resolving it

EVALUATION FOCUS AREAS:
1. **Question Recognition**: Agent identifies and acknowledges all client questions
2. **Response Completeness**: Answers fully address what was asked
3. **Accuracy**: Information provided is correct and reliable
4. **Clarity**: Responses are easy to understand and well-explained
5. **Active Listening**: Agent demonstrates understanding of client's concerns
6. **Proactive Support**: Anticipates needs and offers additional helpful information
7. **Patience**: Shows willingness to explain and re-explain as needed
8. **Concern Resolution**: Addresses underlying worries, not just surface questions

Pay special attention to Ukrainian banking context and Kredobank's customer service standards as evidenced in the transcript.
""".strip()

    json_schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer", 
                "minimum": 0, 
                "maximum": 10,
                "description": "Score for addressing questions (0-10 points)"
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
Evaluate ONLY how well the agent addressed client questions and concerns in this Kredobank collections call transcript:

Transcript:
\"\"\"
{transcript_text.strip()}
\"\"\"

Focus exclusively on the agent's responsiveness to client questions, completeness of answers, and effectiveness in addressing concerns. Ignore all other aspects of the call such as opening procedures, negotiation tactics, or overall professionalism.
"""
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "addressing_questions_evaluation",
                    "schema": json_schema,
                    "strict": True
                }
            }
        )

        result = json.loads(response.choices[0].message.content)
        return result["score"]  # Return just the score number

    except Exception as e:
        return {
            "error": "Addressing questions evaluation failed",
            "details": str(e),
            "raw_response": response.choices[0].message.content if 'response' in locals() else None
        }

def evaluate_negotiation_and_payment_handling(transcript_text: str, model: str = openai_model) -> int:
    """
    Evaluates the negotiation skills and payment handling effectiveness in a Kredobank collections call.
    Focuses on collection strategies, payment arrangements, and achieving concrete payment commitments.
    
    Scoring: 0-20 points
    """
    print(f"Running evaluation for evaluate_negotiation_and_payment_handling")

    SYSTEM_PROMPT = """
You are a specialist QA analyst focusing exclusively on negotiation and payment handling for Kredobank's collections department.

Evaluate ONLY the negotiation and payment handling aspects of the call transcript according to these specific criteria:

NEGOTIATION AND PAYMENT HANDLING SCORING (0-20 points total):

**EXCELLENT (18-20 points):**
- Agent demonstrates excellent negotiation skills and payment-focused approach
- Successfully secures specific payment commitment with clear dates and amounts
- Uses effective persuasion techniques while maintaining respectful tone
- Offers appropriate payment solutions (restructuring, partial payments, etc.)
- Handles payment objections skillfully and provides viable alternatives
- Creates urgency appropriately without being aggressive
- Confirms all payment details and expectations clearly
- Achieves optimal collection outcome for the situation

**GOOD (14-17 points):**
- Agent shows solid negotiation skills with good payment focus
- Secures payment commitment with mostly clear terms
- Uses generally effective persuasion techniques
- Offers some payment solutions when appropriate
- Handles most payment objections adequately
- Creates some urgency while remaining professional
- Confirms most payment details
- Achieves good collection outcome

**ACCEPTABLE (10-13 points):**
- Agent demonstrates basic negotiation skills
- Secures some form of payment commitment, though may lack specificity
- Uses standard persuasion approaches with mixed effectiveness
- Limited offering of payment solutions
- Handles basic payment objections but may miss opportunities
- Creates minimal urgency or pressure
- Confirms some payment details
- Achieves acceptable collection outcome

**NEEDS IMPROVEMENT (6-9 points):**
- Agent shows weak negotiation skills or lacks payment focus
- Fails to secure clear payment commitment or terms are vague
- Ineffective persuasion techniques or missed opportunities
- Does not offer appropriate payment solutions
- Poor handling of payment objections
- Fails to create appropriate urgency
- Inadequate confirmation of payment arrangements
- Poor collection outcome

**POOR (3-5 points):**
- Agent demonstrates poor negotiation skills
- No meaningful payment commitment secured
- Ineffective or inappropriate persuasion attempts
- No payment solutions offered when needed
- Cannot handle payment objections effectively
- Either too aggressive or completely passive
- No proper confirmation of arrangements
- Very poor collection outcome

**UNACCEPTABLE (0-2 points):**
- Agent shows no negotiation skills or payment focus
- Completely fails to secure any payment commitment
- No effective persuasion or collection techniques used
- Makes no effort to arrange payments or offer solutions
- Cannot address payment issues at all
- Inappropriate approach (too aggressive or completely ineffective)
- No follow-up or confirmation attempts
- Complete failure to achieve collection objectives

EVALUATION FOCUS AREAS:
1. **Payment Commitment**: Securing specific dates, amounts, and payment methods
2. **Negotiation Skills**: Effective persuasion, objection handling, and compromise
3. **Solution Offering**: Restructuring options, partial payments, alternative arrangements
4. **Urgency Creation**: Appropriate pressure without aggression
5. **Objection Management**: Addressing client concerns about payment ability
6. **Alternative Strategies**: Offering viable options when initial approach fails
7. **Confirmation Process**: Ensuring clarity on agreed payment terms
8. **Collection Outcome**: Overall effectiveness in achieving payment results
9. **Professional Balance**: Maintaining firm but respectful collection approach

Pay special attention to Ukrainian banking regulations, Kredobank's collection policies, and the balance between firmness and customer relationship preservation as evidenced in the transcript.
""".strip()

    json_schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer", 
                "minimum": 0, 
                "maximum": 20,
                "description": "Score for negotiation and payment handling (0-20 points)"
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
Evaluate ONLY the negotiation and payment handling effectiveness in this Kredobank collections call transcript:

Transcript:
\"\"\"
{transcript_text.strip()}
\"\"\"

Focus exclusively on how well the agent negotiated payment terms, handled payment-related objections, secured payment commitments, and achieved collection objectives. Ignore all other aspects of the call such as opening procedures, general communication clarity, or overall professionalism unless directly related to payment negotiation.
"""
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "negotiation_payment_evaluation",
                    "schema": json_schema,
                    "strict": True
                }
            }
        )

        result = json.loads(response.choices[0].message.content)
        return result["score"]  # Return just the score number

    except Exception as e:
        return {
            "error": "Negotiation and payment handling evaluation failed",
            "details": str(e),
            "raw_response": response.choices[0].message.content if 'response' in locals() else None
        }

def evaluate_professionalism_and_tone(transcript_text: str, model: str = openai_model) -> int:
    """
    Evaluates the professionalism and tone maintained throughout a Kredobank collections call.
    Focuses on brand representation, appropriate communication style, and emotional intelligence.
    
    Scoring: 0-15 points
    """
    print(f"Running evaluation for evaluate_professionalism_and_tone")

    SYSTEM_PROMPT = """
You are a specialist QA analyst focusing exclusively on professionalism and tone for Kredobank's collections department.

Evaluate ONLY the professionalism and tone aspects of the call transcript according to these specific criteria:

PROFESSIONALISM AND TONE SCORING (0-15 points total):

**EXCELLENT (14-15 points):**
- Agent maintains consistently professional, courteous, and respectful tone throughout
- Demonstrates excellent emotional intelligence and empathy when appropriate
- Uses appropriate language register for banking/financial services
- Shows patience and understanding while remaining firm when necessary
- Represents Kredobank brand values excellently
- Handles difficult moments with grace and professionalism
- Maintains appropriate balance between authority and customer service
- Demonstrates cultural sensitivity and appropriate communication style

**GOOD (11-13 points):**
- Agent maintains generally professional and respectful tone
- Shows good emotional intelligence with minor lapses
- Uses mostly appropriate language for banking context
- Demonstrates patience with occasional signs of impatience
- Represents Kredobank brand well overall
- Handles most difficult moments professionally
- Good balance between firmness and courtesy
- Generally appropriate communication style

**ACCEPTABLE (8-10 points):**
- Agent maintains basic professional standards
- Shows adequate emotional intelligence but may miss cues
- Uses acceptable language though may have minor issues
- Shows some patience but may rush occasionally
- Adequate brand representation with room for improvement
- Handles difficult moments adequately but not smoothly
- Reasonable balance though may lean too firm or too soft
- Communication style is functional but not optimal

**NEEDS IMPROVEMENT (5-7 points):**
- Agent shows inconsistent professionalism with notable lapses
- Poor emotional intelligence, misses important emotional cues
- Language choices sometimes inappropriate for banking context
- Shows impatience or frustration that affects tone
- Poor brand representation with multiple concerns
- Struggles with difficult moments, becomes defensive or harsh
- Poor balance, either too aggressive or too passive
- Communication style creates unnecessary tension

**POOR (2-4 points):**
- Agent frequently unprofessional, rude, or inappropriate
- Very poor emotional intelligence, insensitive to client needs
- Uses inappropriate language or unprofessional expressions
- Clearly impatient, frustrated, or dismissive
- Damages Kredobank brand reputation
- Handles difficult moments very poorly, escalates tensions
- Completely inappropriate balance, either hostile or ineffective
- Communication style alienates client

**UNACCEPTABLE (0-1 points):**
- Agent is consistently unprofessional, rude, or hostile
- No emotional intelligence, completely insensitive
- Uses completely inappropriate or offensive language
- Openly hostile, impatient, or disrespectful
- Severely damages Kredobank brand reputation
- Makes difficult situations much worse
- Completely inappropriate approach
- Communication style is harmful to client relationship

EVALUATION FOCUS AREAS:
1. **Tone Consistency**: Maintaining appropriate professional tone throughout call
2. **Emotional Intelligence**: Reading client emotions and responding appropriately
3. **Language Appropriateness**: Using suitable vocabulary and expressions for banking
4. **Patience and Understanding**: Showing appropriate patience with client concerns
5. **Brand Representation**: Reflecting Kredobank's values and professional standards
6. **Conflict Management**: Handling tense moments with professionalism
7. **Authority Balance**: Being firm when needed while remaining respectful
8. **Cultural Sensitivity**: Appropriate communication style for Ukrainian banking context
9. **Courtesy and Respect**: Maintaining polite and respectful demeanor
10. **Stress Management**: Keeping personal frustration from affecting client interaction

Pay special attention to Ukrainian cultural norms, banking industry standards, and Kredobank's commitment to professional customer service even in collections contexts.
""".strip()

    json_schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer", 
                "minimum": 0, 
                "maximum": 15,
                "description": "Score for professionalism and tone (0-15 points)"
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
Evaluate ONLY the professionalism and tone maintained throughout this Kredobank collections call transcript:

Transcript:
\"\"\"
{transcript_text.strip()}
\"\"\"

Focus exclusively on how professionally the agent conducted themselves, the appropriateness of their tone, emotional intelligence displayed, and how well they represented the Kredobank brand. Ignore all other aspects of the call such as technical accuracy, negotiation outcomes, or procedural compliance unless they directly impact professionalism and tone.
"""
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "professionalism_tone_evaluation",
                    "schema": json_schema,
                    "strict": True
                }
            }
        )

        result = json.loads(response.choices[0].message.content)
        return result["score"]  # Return just the score number

    except Exception as e:
        return {
            "error": "Professionalism and tone evaluation failed",
            "details": str(e),
            "raw_response": response.choices[0].message.content if 'response' in locals() else None
        }

def evaluate_call_outcome_and_documentation(transcript_text: str, model: str = openai_model) -> int:
    """
    Evaluates the call outcome achievement and documentation requirements in a Kredobank collections call.
    Focuses on call resolution, follow-up arrangements, and proper call closure with documented next steps.
    
    Scoring: 0-10 points
    """
    print(f"Running evaluation for evaluate_call_outcome_and_documentation")

    SYSTEM_PROMPT = """
You are a specialist QA analyst focusing exclusively on call outcome and documentation for Kredobank's collections department.

Evaluate ONLY the call outcome and documentation aspects of the call transcript according to these specific criteria:

CALL OUTCOME AND DOCUMENTATION SCORING (0-10 points total):

**EXCELLENT (9-10 points):**
- Call achieves clear, measurable outcome with specific next steps defined
- Agent documents all key information and agreements during the call
- Establishes concrete follow-up timeline with specific dates and expectations
- Summarizes key points and confirms mutual understanding before ending
- Provides client with clear action items and contact information if needed
- Call closure is professional and leaves no ambiguity about next steps
- Agent demonstrates understanding of documentation requirements
- All commitments and agreements are clearly stated and confirmed

**GOOD (7-8 points):**
- Call achieves generally clear outcome with mostly defined next steps
- Agent documents most key information and agreements
- Establishes follow-up timeline with reasonably specific expectations
- Provides some summary and confirmation of understanding
- Gives client most necessary action items and information
- Call closure is adequate with minor ambiguity about next steps
- Shows good understanding of documentation needs
- Most commitments are clearly stated

**ACCEPTABLE (5-6 points):**
- Call achieves basic outcome but next steps may lack some clarity
- Agent documents essential information but may miss some details
- Establishes general follow-up expectations but lacks precision
- Provides minimal summary with limited confirmation
- Gives client basic action items but may lack completeness
- Call closure is functional but could be clearer
- Shows adequate understanding of documentation requirements
- Basic commitments are stated but may need more detail

**NEEDS IMPROVEMENT (3-4 points):**
- Call outcome is unclear or poorly defined
- Agent fails to document several important details
- Follow-up arrangements are vague or incomplete
- Little to no summary or confirmation of understanding
- Client left with unclear action items or expectations
- Call closure is abrupt or confusing
- Poor understanding of documentation requirements
- Commitments are poorly defined or not confirmed

**POOR (1-2 points):**
- Call achieves no clear outcome or resolution
- Agent documents very little or provides inaccurate information
- No proper follow-up arrangements established
- No summary or confirmation attempted
- Client has no clear understanding of next steps
- Call ends without proper closure
- No evidence of documentation awareness
- No clear commitments or agreements established

**UNACCEPTABLE (0 points):**
- Call completely fails to achieve any meaningful outcome
- No documentation or completely inaccurate information
- No follow-up arrangements or next steps established
- Call ends abruptly without resolution or explanation
- Client is left completely confused about expectations
- No professional call closure
- Complete failure to meet documentation requirements
- No commitments, agreements, or clear path forward

EVALUATION FOCUS AREAS:
1. **Outcome Achievement**: Clear resolution or defined next steps from the call
2. **Information Documentation**: Recording key details, agreements, and commitments
3. **Follow-up Planning**: Establishing specific timelines and expectations for next contact
4. **Call Summary**: Reviewing key points and ensuring mutual understanding
5. **Client Action Items**: Clearly communicating what client needs to do next
6. **Confirmation Process**: Ensuring both parties understand agreements and next steps
7. **Professional Closure**: Ending call appropriately with clear expectations
8. **Documentation Completeness**: Capturing all necessary information for records
9. **Commitment Clarity**: Ensuring all promises and agreements are explicit
10. **Process Compliance**: Following proper procedures for call completion

Pay special attention to Ukrainian banking documentation standards, Kredobank's collection procedures, and regulatory requirements for call records and follow-up processes.
""".strip()

    json_schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer", 
                "minimum": 0, 
                "maximum": 10,
                "description": "Score for call outcome and documentation (0-10 points)"
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
Evaluate ONLY the call outcome and documentation aspects of this Kredobank collections call transcript:

Transcript:
\"\"\"
{transcript_text.strip()}
\"\"\"

Focus exclusively on how well the agent achieved a clear call outcome, documented important information, established follow-up procedures, and properly closed the call with clear next steps. Ignore all other aspects of the call such as opening procedures, negotiation techniques, or tone unless they directly impact the call outcome and documentation process.
"""
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "call_outcome_documentation_evaluation",
                    "schema": json_schema,
                    "strict": True
                }
            }
        )

        result = json.loads(response.choices[0].message.content)
        return result["score"]  # Return just the score number

    except Exception as e:
        return {
            "error": "Call outcome and documentation evaluation failed",
            "details": str(e),
            "raw_response": response.choices[0].message.content if 'response' in locals() else None
        }
    
def evaluate_compliance(transcript_text: str, model: str = openai_model) -> str:
    """
    Evaluates compliance with legal and regulatory requirements in a Kredobank collections call.
    This is a critical pass/fail evaluation that determines if the call meets mandatory standards.
    
    Returns: "PASS" or "FAIL" (no points - critical gate)
    """
    print(f"Running evaluation for evaluate_compliance")

    SYSTEM_PROMPT = """
You are a specialist compliance analyst focusing exclusively on legal and regulatory compliance for Kredobank's collections department.

Evaluate ONLY the compliance aspects of the call transcript according to these critical requirements:

COMPLIANCE EVALUATION (PASS/FAIL - NO SCORING):

**AUTOMATIC FAIL CONDITIONS (any one of these results in FAIL):**
- No call recording disclosure provided
- No proper agent identification (name and department)
- No client identity verification attempted
- Use of threatening, abusive, or inappropriate language
- Violation of debt collection laws or banking regulations
- Harassment or excessive pressure tactics
- Disclosure of debt information to unauthorized parties
- False or misleading statements about consequences
- Attempts to collect amounts not legally owed
- Violation of contact time restrictions or frequency limits
- Failure to respect client's stated preferences about contact
- Any behavior that could constitute unfair debt collection practices

**PASS REQUIREMENTS (ALL must be met for PASS):**
1. **Recording Disclosure**: Clear statement that call is being recorded
2. **Agent Identification**: Agent provides name and identifies Kredobank/department
3. **Identity Verification**: Attempt to verify client identity (name confirmation)
4. **Appropriate Language**: Professional, respectful language throughout
5. **Legal Compliance**: No violations of Ukrainian debt collection laws
6. **Truth in Communication**: All statements are accurate and not misleading
7. **Respectful Treatment**: Client treated with dignity and respect
8. **Proper Contact**: Call made at appropriate time and manner
9. **Accurate Information**: Debt amounts and details are correctly stated
10. **Regulatory Adherence**: Follows banking industry regulations and standards

**CRITICAL COMPLIANCE AREAS:**
- **Ukrainian Debt Collection Laws**: Must comply with local regulations
- **Banking Regulations**: Adherence to National Bank of Ukraine requirements  
- **Consumer Protection**: Respecting client rights and protections
- **Data Protection**: Proper handling of personal and financial information
- **Fair Treatment**: No discriminatory or abusive practices
- **Truthfulness**: All communications must be accurate and honest
- **Professional Standards**: Maintaining appropriate business conduct

**EVALUATION APPROACH:**
- Look for any violations that would constitute regulatory non-compliance
- Assess whether the call could expose Kredobank to legal liability
- Determine if client rights were respected throughout the interaction
- Verify that all mandatory disclosure and identification requirements were met
- Check for any practices that violate fair debt collection standards

**IMPORTANT:** This is a binary evaluation. Even excellent performance in other areas cannot overcome a compliance failure. One significant compliance violation results in an overall FAIL rating.
""".strip()

    json_schema = {
        "type": "object",
        "properties": {
            "compliance_result": {
                "type": "string",
                "enum": ["PASS", "FAIL"],
                "description": "Compliance evaluation result - PASS or FAIL"
            },
            "violation_details": {
                "type": "string",
                "description": "Details of any compliance violations found (if FAIL) or confirmation of compliance (if PASS)"
            }
        },
        "required": ["compliance_result", "violation_details"],
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
Evaluate ONLY the compliance aspects of this Kredobank collections call transcript:

Transcript:
\"\"\"
{transcript_text.strip()}
\"\"\"

Determine if this call PASSES or FAILS compliance requirements. Focus exclusively on legal and regulatory compliance, mandatory disclosures, proper procedures, and adherence to debt collection laws. This is a critical gate - any significant compliance violation results in FAIL regardless of other call qualities.

Provide the compliance result (PASS or FAIL) and detailed explanation of your evaluation.
"""
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "compliance_evaluation",
                    "schema": json_schema,
                    "strict": True
                }
            }
        )

        result = json.loads(response.choices[0].message.content)
        return result["compliance_result"]  # Return just the PASS/FAIL result

    except Exception as e:
        return {
            "error": "Compliance evaluation failed",
            "details": str(e),
            "raw_response": response.choices[0].message.content if 'response' in locals() else None
        }

def evaluate_call_efficiency_and_focus(transcript_text: str, model: str = openai_model) -> int:
    """
    Evaluates the efficiency and focus of a Kredobank collections call.
    Focuses on time management, staying on topic, achieving objectives efficiently, and resource optimization.
    
    Scoring: 0-10 points
    """
    print(f"Running evaluation for evaluate_call_efficiency_and_focus")

    SYSTEM_PROMPT = """
You are a specialist QA analyst focusing exclusively on call efficiency and focus for Kredobank's collections department.

Evaluate ONLY the efficiency and focus aspects of the call transcript according to these specific criteria:

CALL EFFICIENCY AND FOCUS SCORING (0-10 points total):

**EXCELLENT (9-10 points):**
- Call stays consistently focused on collection objectives throughout
- Agent manages time efficiently with no unnecessary delays or tangents
- Moves systematically through required steps without redundancy
- Achieves maximum results with minimal time investment
- Maintains productive pace while allowing adequate time for client responses
- Efficiently handles objections and redirects conversations back to objectives
- Uses time-saving techniques and streamlined approaches
- Demonstrates excellent resource management and productivity

**GOOD (7-8 points):**
- Call maintains good focus on collection objectives with minor diversions
- Agent manages time well with minimal inefficiencies
- Follows structured approach with occasional redundancy
- Achieves good results with reasonable time investment
- Generally productive pace with appropriate client interaction time
- Handles most objections efficiently with some delays
- Uses some time-saving techniques effectively
- Shows good resource management overall

**ACCEPTABLE (5-6 points):**
- Call maintains basic focus but may have some unnecessary diversions
- Agent manages time adequately but misses some efficiency opportunities
- Follows general structure but with noticeable redundancy or gaps
- Achieves acceptable results but could be more time-efficient
- Reasonable pace but may rush or drag at times
- Handles objections adequately but not always efficiently
- Limited use of time-saving techniques
- Shows adequate resource management with room for improvement

**NEEDS IMPROVEMENT (3-4 points):**
- Call loses focus frequently with multiple unnecessary tangents
- Poor time management with significant inefficiencies
- Disorganized approach with considerable redundancy
- Limited results relative to time invested
- Poor pacing - either too rushed or unnecessarily slow
- Inefficient handling of objections, gets sidetracked easily
- Fails to use available time-saving techniques
- Poor resource management and productivity

**POOR (1-2 points):**
- Call frequently off-topic with little focus on collection objectives
- Very poor time management with major inefficiencies
- Highly disorganized with excessive redundancy and confusion
- Minimal results for significant time investment
- Very poor pacing that impedes effectiveness
- Cannot handle objections efficiently, easily derailed
- No evidence of time-saving or efficiency techniques
- Very poor resource utilization

**UNACCEPTABLE (0 points):**
- Call completely lacks focus on collection objectives
- Extremely poor time management, highly inefficient
- Completely disorganized with no clear structure
- No meaningful results achieved relative to time spent
- Completely inappropriate pacing
- Cannot manage objections or stay on track
- No efficiency awareness or techniques used
- Complete waste of resources and time

EVALUATION FOCUS AREAS:
1. **Objective Focus**: Staying on track with collection goals throughout call
2. **Time Management**: Efficient use of call time without unnecessary delays
3. **Structured Approach**: Following logical sequence without redundancy
4. **Pace Management**: Appropriate speed that maximizes efficiency while allowing client participation
5. **Distraction Handling**: Quickly redirecting off-topic conversations back to objectives
6. **Objection Efficiency**: Handling client concerns without getting sidetracked
7. **Resource Optimization**: Making best use of agent time and company resources
8. **Productivity Measures**: Achieving maximum results with minimal time investment
9. **Process Streamlining**: Using efficient techniques and avoiding unnecessary steps
10. **Call Length Appropriateness**: Achieving objectives in reasonable timeframe

Pay special attention to Kredobank's productivity expectations, call center efficiency standards, and the balance between thorough service and resource management.
""".strip()

    json_schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer", 
                "minimum": 0, 
                "maximum": 10,
                "description": "Score for call efficiency and focus (0-10 points)"
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
Evaluate ONLY the efficiency and focus aspects of this Kredobank collections call transcript:

Transcript:
\"\"\"
{transcript_text.strip()}
\"\"\"

Focus exclusively on how efficiently the agent managed the call time, stayed focused on collection objectives, handled distractions, and optimized resource usage. Ignore all other aspects of the call such as tone, compliance, or negotiation outcomes unless they directly impact call efficiency and focus.
"""
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "call_efficiency_focus_evaluation",
                    "schema": json_schema,
                    "strict": True
                }
            }
        )

        result = json.loads(response.choices[0].message.content)
        return result["score"]  # Return just the score number

    except Exception as e:
        return {
            "error": "Call efficiency and focus evaluation failed",
            "details": str(e),
            "raw_response": response.choices[0].message.content if 'response' in locals() else None
        }

def evaluate_handling_sensitive_situations(transcript_text: str, model: str = openai_model) -> int:
    """
    Evaluates how well the agent handles sensitive situations in a Kredobank collections call.
    Focuses on empathy, crisis management, special circumstances, and appropriate responses to client distress.
    
    Scoring: 0-10 points
    """
    print(f"Running evaluation for evaluate_handling_sensitive_situations")

    SYSTEM_PROMPT = """
You are a specialist QA analyst focusing exclusively on handling sensitive situations for Kredobank's collections department.

Evaluate ONLY how well the agent handles sensitive situations in the call transcript according to these specific criteria:

HANDLING SENSITIVE SITUATIONS SCORING (0-10 points total):

**EXCELLENT (9-10 points):**
- Agent demonstrates exceptional empathy and emotional intelligence when client shares difficulties
- Responds appropriately to crisis situations (job loss, medical issues, family emergencies, war-related impacts)
- Shows genuine understanding and validates client's emotional state
- Offers appropriate solutions and alternatives for difficult circumstances
- Maintains professional boundaries while showing human compassion
- De-escalates tense situations skillfully and calmly
- Adapts communication style appropriately for vulnerable clients
- Demonstrates cultural sensitivity and awareness of current events (e.g., war impact)

**GOOD (7-8 points):**
- Agent shows good empathy and responds well to client difficulties
- Handles most crisis situations appropriately with minor gaps
- Generally validates client emotions and shows understanding
- Offers some appropriate solutions for difficult circumstances
- Maintains mostly appropriate professional boundaries
- De-escalates most tense situations effectively
- Shows some adaptation in communication style
- Generally sensitive to cultural and situational context

**ACCEPTABLE (5-6 points):**
- Agent shows basic empathy but may miss some emotional cues
- Handles crisis situations adequately but could be more supportive
- Provides minimal validation of client emotions
- Offers limited solutions for difficult circumstances
- Professional boundaries mostly maintained but may lack warmth
- Basic de-escalation skills with mixed effectiveness
- Limited adaptation of communication style
- Some awareness of cultural/situational sensitivity

**NEEDS IMPROVEMENT (3-4 points):**
- Agent shows little empathy or misses important emotional cues
- Poor handling of crisis situations, lacks appropriate response
- Fails to validate client emotions or acknowledge difficulties
- Does not offer helpful solutions for challenging circumstances
- Poor professional boundaries, either too cold or inappropriate
- Weak de-escalation skills, may increase tensions
- No adaptation of communication style for sensitive situations
- Little awareness of cultural or situational sensitivity

**POOR (1-2 points):**
- Agent shows no empathy or makes situation worse
- Completely inappropriate response to crisis situations
- Dismissive of client emotions and difficulties
- No solutions offered for challenging circumstances
- Inappropriate professional boundaries
- Cannot de-escalate, often makes situations worse
- Completely inflexible communication approach
- No cultural or situational sensitivity

**UNACCEPTABLE (0 points):**
- Agent is hostile or harmful when client shares difficulties
- Extremely inappropriate handling of crisis situations
- Completely dismissive or cruel regarding client emotions
- Actively unhelpful with difficult circumstances
- Completely inappropriate boundaries
- Escalates rather than de-escalates sensitive situations
- Harmful or inappropriate communication
- Culturally insensitive or offensive

EVALUATION FOCUS AREAS:
1. **Empathy Recognition**: Identifying when client is experiencing distress or difficulties
2. **Emotional Validation**: Acknowledging and validating client's emotional state
3. **Crisis Response**: Appropriate handling of emergency or crisis situations
4. **Solution Orientation**: Offering helpful alternatives for challenging circumstances
5. **De-escalation Skills**: Calming tense or emotional situations
6. **Professional Compassion**: Balancing business objectives with human understanding
7. **Communication Adaptation**: Adjusting approach for vulnerable or distressed clients
8. **Cultural Sensitivity**: Understanding context of Ukrainian situation, war impact, economic challenges
9. **Boundary Management**: Maintaining professional limits while showing care
10. **Special Circumstances**: Recognizing and appropriately handling unique situations

**SPECIAL CIRCUMSTANCES TO WATCH FOR:**
- War-related impacts (job loss, displacement, economic hardship)
- Medical emergencies or health crises
- Family emergencies or bereavements
- Job loss or income reduction
- Mental health concerns or emotional distress
- Elderly or vulnerable clients
- Language barriers or communication difficulties
- Financial literacy issues

Pay special attention to the current Ukrainian context, including war impacts, economic challenges, and the need for agents to balance collection objectives with human compassion during difficult times.
""".strip()

    json_schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer", 
                "minimum": 0, 
                "maximum": 10,
                "description": "Score for handling sensitive situations (0-10 points)"
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
Evaluate ONLY how well the agent handled sensitive situations in this Kredobank collections call transcript:

Transcript:
\"\"\"
{transcript_text.strip()}
\"\"\"

Focus exclusively on the agent's empathy, emotional intelligence, crisis management, and ability to handle special circumstances or client distress appropriately. Look for moments where the client shared difficulties, expressed emotions, or presented challenging situations, and evaluate how well the agent responded. 

If no particularly sensitive situations arose in this call, evaluate based on the agent's general readiness to handle such situations and any minor emotional cues that may have been present.

Ignore all other aspects of the call such as technical accuracy, compliance, or negotiation outcomes unless they directly relate to handling sensitive situations.
"""
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sensitive_situations_evaluation",
                    "schema": json_schema,
                    "strict": True
                }
            }
        )

        result = json.loads(response.choices[0].message.content)
        return result["score"]  # Return just the score number

    except Exception as e:
        return {
            "error": "Handling sensitive situations evaluation failed",
            "details": str(e),
            "raw_response": response.choices[0].message.content if 'response' in locals() else None
        }



def evaluate_overall_analysis(transcript_text: str, evaluation_results: dict, model: str = openai_model) -> dict:
    """
    Creates an overall evaluation analysis with detailed commentary based primarily on the call transcript.
    Provides strengths, areas for improvement, and compliance notes.
    
    Returns: dict with strengths, areas_for_improvement, and compliance_notes
    """
    print(f"Running overall evaluation analysis")

    SYSTEM_PROMPT = """
Ви старший QA аналітик відділу стягнення заборгованості Кредобанку, який надає стислий зворотний зв'язок щодо оцінки.

На основі переважно транскрипту дзвінка надайте коротний коментар у трьох областях. Кожна відповідь має бути максимум 2-3 речення.

**АНАЛІЗ СИЛЬНИХ СТОРІН (2-3 речення):**
Визначте найважливіші речі, які агент зробив добре, зосереджуючись на конкретних прикладах з транскрипту.

**АНАЛІЗ ОБЛАСТЕЙ ДЛЯ ПОКРАЩЕННЯ (2-3 речення):**
Визначте найкритичніші області, що потребують розвитку, з однією конкретною практичною рекомендацією.

**АНАЛІЗ НОТАТОК ЩОДО ДОТРИМАННЯ ВИМОГ (2-3 речення):**
Оцініть дотримання регулятивних вимог, зосереджуючись на обов'язкових розкриттях, верифікації особистості та професійній поведінці.

**ВИМОГИ:**
- Максимум 2-3 речення на розділ
- Зосередьтеся лише на найважливіших моментах
- Використовуйте конкретні приклади з транскрипту
- Надавайте практичний, дієвий зворотний зв'язок
- Враховуйте контекст українського банкінгу
- ВАЖЛИВО: Всі відповіді мають бути українською мовою
""".strip()

    json_schema = {
        "type": "object",
        "properties": {
            "strengths": {
                "type": "string",
                "description": "Detailed analysis of what the agent did well, with specific examples from the transcript"
            },
            "areas_for_improvement": {
                "type": "string", 
                "description": "Specific opportunities for enhancement with actionable recommendations"
            },
            "compliance_notes": {
                "type": "string",
                "description": "Assessment of regulatory compliance, procedural adherence, and related recommendations"
            }
        },
        "required": ["strengths", "areas_for_improvement", "compliance_notes"],
        "additionalProperties": False
    }

    try:
        response = openai_client.chat.completions.create(
            model=model,
            temperature=0.3,  # Slightly higher temperature for more nuanced analysis
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT                          
                },
                {
                    "role": "user",
                    "content": f"""
Проведіть комплексний аналіз оцінки цього дзвінка стягнення заборгованості Кредобанку.

ТРАНСКРИПТ ДЗВІНКА:
\"\"\"
{transcript_text.strip()}
\"\"\"

РЕЗУЛЬТАТИ ОЦІНКИ (для довідки):
- Відкриття та Верифікація: {evaluation_results.get('opening_and_verification', 'Н/Д')}/10
- Чіткість та Точність: {evaluation_results.get('clarity_and_accuracy', 'Н/Д')}/15
- Відповіді на Питання: {evaluation_results.get('addressing_questions', 'Н/Д')}/10
- Переговори та Обробка Платежів: {evaluation_results.get('negotiation_and_payment_handling', 'Н/Д')}/20
- Професійність та Тон: {evaluation_results.get('professionalism_and_tone', 'Н/Д')}/15
- Результат Дзвінка та Документування: {evaluation_results.get('call_outcome_and_documentation', 'Н/Д')}/10
- Статус Дотримання Вимог: {evaluation_results.get('compliance_passed', 'Н/Д')}
- Ефективність та Фокус Дзвінка: {evaluation_results.get('call_efficiency_and_focus', 'Н/Д')}/10
- Обробка Деліkatних Ситуацій: {evaluation_results.get('handling_sensitive_situations', 'Н/Д')}/10
- Загальний Бал: {evaluation_results.get('total_score', 'Н/Д')}/100

ОСНОВНА ІНСТРУКЦІЯ: Базуйте свій аналіз переважно на фактичному змісті транскрипту. Обмежте кожен розділ рівно 2-3 реченнями. Зосередьтеся лише на найважливіших моментах.

Надайте стислий коментар для:
1. СИЛЬНІ СТОРОНИ: Що агент зробив виключно добре? (2-3 речення українською)
2. ОБЛАСТІ ДЛЯ ПОКРАЩЕННЯ: Яка найкритичніша область, що потребує розвитку? (2-3 речення українською)
3. НОТАТКИ ЩОДО ДОТРИМАННЯ ВИМОГ: Наскільки добре агент виконав регулятивні вимоги? (2-3 речення українською)
"""
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "overall_evaluation_analysis",
                    "schema": json_schema,
                    "strict": True
                }
            }
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        return {
            "error": "Помилка аналізу загальної оцінки",
            "details": str(e),
            "raw_response": response.choices[0].message.content if 'response' in locals() else None
        }


def evaluate_call(transcript_text: str, model: str = openai_model) -> dict:
    """
    Complete wrapper function that runs all evaluation functions and creates overall analysis.
    """
    
    evaluation_results = {
        "opening_and_verification": None,           # 10 points - Critical foundation
        "clarity_and_accuracy": None,              # 15 points - Core communication
        "addressing_questions": None,              # 10 points - Client service
        "negotiation_and_payment_handling": None,  # 20 points - Primary objective
        "professionalism_and_tone": None,         # 15 points - Brand representation
        "call_outcome_and_documentation": None,    # 10 points - Process completion
        "compliance_passed": None,                 # Pass/Fail - No points (critical gate)
        "call_efficiency_and_focus": None,        # 10 points - Resource management
        "handling_sensitive_situations": None,     # 10 points - Special circumstances
        "total_score": 0,
        
        # NEW: Overall analysis fields
        "strengths": None,                          # Detailed strengths analysis
        "areas_for_improvement": None,              # Improvement recommendations  
        "compliance_notes": None                    # Compliance assessment
    }
    
    # Run all individual evaluations
    try:
        evaluation_results["opening_and_verification"] = evaluate_opening_and_verification(transcript_text, model)
        evaluation_results["total_score"] += evaluation_results["opening_and_verification"]
    except Exception as e:
        print(f"Error in opening_and_verification: {e}")
    
    try:
        evaluation_results["clarity_and_accuracy"] = evaluate_clarity_and_accuracy(transcript_text, model)
        evaluation_results["total_score"] += evaluation_results["clarity_and_accuracy"]
    except Exception as e:
        print(f"Error in clarity_and_accuracy: {e}")
        
    try:
        evaluation_results["addressing_questions"] = evaluate_addressing_questions(transcript_text, model)
        evaluation_results["total_score"] += evaluation_results["addressing_questions"]
    except Exception as e:
        print(f"Error in addressing_questions: {e}")

    try:
        evaluation_results["negotiation_and_payment_handling"] = evaluate_negotiation_and_payment_handling(transcript_text, model)
        evaluation_results["total_score"] += evaluation_results["negotiation_and_payment_handling"]
    except Exception as e:
        print(f"Error in negotiation_and_payment_handling: {e}")

    try:
        evaluation_results["professionalism_and_tone"] = evaluate_professionalism_and_tone(transcript_text, model)
        evaluation_results["total_score"] += evaluation_results["professionalism_and_tone"]
    except Exception as e:
        print(f"Error in professionalism_and_tone: {e}")

    try:
        evaluation_results["call_outcome_and_documentation"] = evaluate_call_outcome_and_documentation(transcript_text, model)
        evaluation_results["total_score"] += evaluation_results["call_outcome_and_documentation"]
    except Exception as e:
        print(f"Error in call_outcome_and_documentation: {e}")

    try:
        evaluation_results["compliance_passed"] = evaluate_compliance(transcript_text, model)
        # Note: No points added to total_score as this is Pass/Fail only
    except Exception as e:
        print(f"Error in compliance evaluation: {e}")
        evaluation_results["compliance_passed"] = "ERROR"

    try:
        evaluation_results["call_efficiency_and_focus"] = evaluate_call_efficiency_and_focus(transcript_text, model)
        evaluation_results["total_score"] += evaluation_results["call_efficiency_and_focus"]
    except Exception as e:
        print(f"Error in call_efficiency_and_focus: {e}")

    try:
        evaluation_results["handling_sensitive_situations"] = evaluate_handling_sensitive_situations(transcript_text, model)
        evaluation_results["total_score"] += evaluation_results["handling_sensitive_situations"]
    except Exception as e:
        print(f"Error in handling_sensitive_situations: {e}")

    # NEW: Run overall analysis after all individual evaluations
    try:
        overall_analysis = evaluate_overall_analysis(transcript_text, evaluation_results, model)
        evaluation_results["strengths"] = overall_analysis.get("strengths")
        evaluation_results["areas_for_improvement"] = overall_analysis.get("areas_for_improvement")
        evaluation_results["compliance_notes"] = overall_analysis.get("compliance_notes")
    except Exception as e:
        print(f"Error in overall analysis: {e}")
        evaluation_results["strengths"] = "Analysis unavailable due to error"
        evaluation_results["areas_for_improvement"] = "Analysis unavailable due to error"
        evaluation_results["compliance_notes"] = "Analysis unavailable due to error"

    return evaluation_results



def format_evaluation_results(evaluation_results: dict) -> str:
    """
    Enhanced format function to create formatted evaluation results with detailed analysis.
    Returns formatted string instead of printing.
    """
    
    output = []
    
    output.append("=" * 70)
    output.append("РЕЗУЛЬТАТИ ОЦІНКИ ДЗВІНКА КРЕДОБАНК")
    output.append("=" * 70)
    
    # Scoring Results
    output.append("\nРОЗБИВКА БАЛІВ:")
    output.append("-" * 40)
    output.append(f"Відкриття та Верифікація: {evaluation_results['opening_and_verification']}/10")
    output.append(f"Чіткість та Точність: {evaluation_results['clarity_and_accuracy']}/15")
    output.append(f"Відповіді на Питання: {evaluation_results['addressing_questions']}/10")
    output.append(f"Переговори та Обробка Платежів: {evaluation_results['negotiation_and_payment_handling']}/20")
    output.append(f"Професійність та Тон: {evaluation_results['professionalism_and_tone']}/15")
    output.append(f"Результат Дзвінка та Документування: {evaluation_results['call_outcome_and_documentation']}/10")
    output.append(f"Дотримання Вимог: {evaluation_results['compliance_passed']}")
    output.append(f"Ефективність та Фокус Дзвінка: {evaluation_results['call_efficiency_and_focus']}/10")
    output.append(f"Обробка Деліkatних Ситуацій: {evaluation_results['handling_sensitive_situations']}/10")
    
    output.append("-" * 40)
    output.append(f"ЗАГАЛЬНИЙ БАЛ: {evaluation_results['total_score']}/100")
    
    # Detailed Analysis
    output.append("\n" + "=" * 70)
    output.append("ДЕТАЛЬНИЙ АНАЛІЗ")
    output.append("=" * 70)
    
    output.append("\nСИЛЬНІ СТОРОНИ:")
    output.append("-" * 20)
    output.append(evaluation_results.get('strengths', 'Аналіз недоступний'))
    
    output.append("\nОБЛАСТІ ДЛЯ ПОКРАЩЕННЯ:")
    output.append("-" * 30)
    output.append(evaluation_results.get('areas_for_improvement', 'Аналіз недоступний'))
    
    output.append("\nНОТАТКИ ЩОДО ДОТРИМАННЯ ВИМОГ:")
    output.append("-" * 20)
    output.append(evaluation_results.get('compliance_notes', 'Аналіз недоступний'))
    
    output.append("\n")
    
    return "\n".join(output)

def format_evaluation_results_md(evaluation_results: dict) -> str:
    """
    Format function to create evaluation results in Markdown format.
    Returns formatted Markdown string.
    """
    
    output = []
    
    # Scoring Results
    output.append("")
    output.append("")
    output.append("#### Розбивка балів")
    output.append("")
    output.append("| Критерій | Бал |")
    output.append("|----------|-----|")
    output.append(f"| Відкриття та Верифікація | {evaluation_results['opening_and_verification']}/10 |")
    output.append(f"| Чіткість та Точність | {evaluation_results['clarity_and_accuracy']}/15 |")
    output.append(f"| Відповіді на Питання | {evaluation_results['addressing_questions']}/10 |")
    output.append(f"| Переговори та Обробка Платежів | {evaluation_results['negotiation_and_payment_handling']}/20 |")
    output.append(f"| Професійність та Тон | {evaluation_results['professionalism_and_tone']}/15 |")
    output.append(f"| Результат Дзвінка та Документування | {evaluation_results['call_outcome_and_documentation']}/10 |")
    output.append(f"| Дотримання Вимог | {evaluation_results['compliance_passed']} |")
    output.append(f"| Ефективність та Фокус Дзвінка | {evaluation_results['call_efficiency_and_focus']}/10 |")
    output.append(f"| Обробка Деліkatних Ситуацій | {evaluation_results['handling_sensitive_situations']}/10 |")
    output.append("")
    output.append(f"#### **Загальний бал: {evaluation_results['total_score']}/100**")
    output.append("")
    
    # Detailed Analysis
    output.append("#### Детальний аналіз")
    output.append("")
    
    output.append("##### Сильні сторони")
    output.append("")
    output.append(evaluation_results.get('strengths', 'Аналіз недоступний'))
    output.append("")
    
    output.append("##### Області для покращення")
    output.append("")
    output.append(evaluation_results.get('areas_for_improvement', 'Аналіз недоступний'))
    output.append("")
    
    output.append("##### Нотатки щодо дотримання вимог")
    output.append("")
    output.append(evaluation_results.get('compliance_notes', 'Аналіз недоступний'))
    output.append("")
    output.append("")
    
    return "\n".join(output)



def print_evaluation_results(evaluation_results_text):
    """
    Print function that uses the format function for backward compatibility.
    """
    print(evaluation_results_text)

    