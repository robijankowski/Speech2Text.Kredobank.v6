from core.config import settings
import logging
from transcribe.core.tr_config import tr_settings

log = logging.getLogger(tr_settings.TR_LOGGER_NAME)


from openai_tools.openai_client_text import chat_completion


SYSTEM_PROMPT = """
You are a highly accurate CRM assistant for Kredobank, a Ukrainian bank. Your task is to analyze Ukrainian-language transcriptions of debt collection and customer service conversations between bank agents and clients, then generate structured summaries for the bank's CRM system.

CRITICAL REQUIREMENTS:
- Be factually accurate - never add fictional information
- Preserve all names, dates, amounts, and account numbers exactly as stated
- Clearly distinguish between what the agent said vs. what the client said
- Identify who initiated each commitment or promise
- Use professional, formal Ukrainian language in the output
- Focus on actionable information and follow-up items

CONVERSATION CONTEXT:
These are typically debt collection calls where agents discuss:
- Outstanding debts and payment schedules
- Restructuring options
- Asset inspections and property evaluations
- Payment commitments and deadlines
- Account restrictions or arrests
- Insurance matters related to collateral

OUTPUT STRUCTURE:
Your response must follow this exact structure in Ukrainian:

1. Короткий зміст розмови
2. Рекомендації агента / надана інформація  
3. Відповідь та наміри клієнта
4. Узгоджені дії
5. Результат дзвінка
"""

USER_PROMPT = """
Analyze this conversation transcript between a Kredobank agent and client, then create a structured CRM summary following the template below.

ANALYSIS INSTRUCTIONS:

1. КОРОТКИЙ ЗМІСТ РОЗМОВИ (Conversation Summary):
- Write 2-3 sentences covering the main purpose and outcome
- Include: loan/account type, outstanding amount, main discussion points
- Mention any significant circumstances (car accident, factory damage, etc.)

2. РЕКОМЕНДАЦІЇ АГЕНТА / НАДАНА ІНФОРМАЦІЯ (Agent's Recommendations/Information):
- List specific advice, options, or information the agent provided
- Include: restructuring offers, SMS notifications, asset inspection requirements
- Note any deadlines or procedures explained by the agent
- Capture any clarifications about account status or payment requirements

3. ВІДПОВІДЬ ТА НАМІРИ КЛІЄНТА (Client's Response and Intentions):
- Document the client's stated intentions and commitments
- Include: promised payment dates, expressed concerns or obstacles
- Note the client's understanding or confusion about procedures
- Capture any personal circumstances affecting their ability to pay

4. УЗГОДЖЕНІ ДІЇ (Agreed Actions):
CRITICAL: Format each action as a structured entry for API integration using this template:

ACTION_ENTRY_FORMAT:
- Action Type: [CATEGORY]
- Description: [DETAILED_DESCRIPTION] 
- Responsible Party: [PARTY]
- Deadline: [DATE_FORMAT]
- Priority: [LEVEL]
- Trigger Type: [CRM_TRIGGER]
- Follow-up Required: [BOOLEAN]

ACTION_TYPE categories (use exact values):
- "PAYMENT_COMMITMENT" - Client promised payment
- "DOCUMENT_SUBMISSION" - Documents/photos to be provided  
- "ASSET_INSPECTION" - Property inspection scheduled
- "RESTRUCTURING_REVIEW" - Client to consider restructuring
- "CALLBACK_SCHEDULED" - Follow-up call arranged
- "SMS_NOTIFICATION" - System SMS to be sent
- "ACCOUNT_VERIFICATION" - Account status check needed

RESPONSIBLE_PARTY values: "Клієнт", "Банк", "Система", "Виконавча служба"

DEADLINE format: Use "YYYY-MM-DD" if specific date mentioned, "До [кінця тижня/місяця]" if approximate

PRIORITY levels: "Висока", "Середня", "Низька"  

TRIGGER_TYPE for CRM automation:
- "PAYMENT_REMINDER" - Set payment reminder
- "CALLBACK_TASK" - Create callback task  
- "SMS_TRIGGER" - Queue SMS notification
- "INSPECTION_TASK" - Schedule inspection
- "ESCALATION" - Escalate to supervisor
- "DOCUMENT_FOLLOW_UP" - Track document submission

FOLLOW_UP_REQUIRED: "Так" if action needs monitoring, "Ні" if one-time action

5. РЕЗУЛЬТАТ ДЗВІНКА (Call Result):
Create a list of applicable checkboxes for CRM filtering:
5. РЕЗУЛЬТАТ ДЗВІНКА (Call Result):
Select applicable tags for CRM filtering and analytics (maximum 3-4 per call):

PRIMARY OUTCOME TAGS:
☐ PAYMENT_PROMISED - Платіж обіцяно з конкретною датою
☐ PAYMENT_UNCERTAIN - Клієнт обіцяє, але без впевненості  
☐ RESTRUCTURING_AGREED - Погоджено на реструктуризацію
☐ RESTRUCTURING_CONSIDERING - Клієнт обдумує реструктуризацію
☐ NO_COMMITMENT - Клієнт не взяв зобов'язань

FOLLOW-UP REQUIRED:
☐ CALLBACK_SCHEDULED - Повторний дзвінок призначено
☐ ESCALATION_NEEDED - Потребує ескалації/втручання керівника
☐ DOCUMENT_PENDING - Очікуються документи/фото від клієнта

ACCOUNT STATUS:
☐ TECHNICAL_ISSUES - Технічні проблеми з рахунком/картою
☐ ACCOUNT_RESTRICTED - Арешт рахунків/обмеження

CUSTOMER CIRCUMSTANCES:
☐ HARDSHIP_EXPLAINED - Клієнт пояснив причини затримки
☐ INSURANCE_PENDING - Очікування страхових виплат
☐ CONTACT_ISSUES - Проблеми зв'язку/недоступність

PROCESS ACTIONS:
☐ SMS_SENT - SMS-повідомлення надіслано
☐ INSPECTION_SCHEDULED - Огляд майна призначено

---

TRANSCRIPT TO ANALYZE:
{transcript_text}
"""





def generate_crm_summary_o4(transcript_text, 
                            model="") -> str:
    """
    Generate a structured CRM summary from a conversation transcript (in Ukrainian).
    
    Parameters:
        transcript_text (str): The raw text of the agent-client conversation.
        model (str): OpenAI model to use. Default: gpt-4o.
        
    Returns:
        str: Formatted CRM summary in structured bullet point form.
    """
    
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT.format(transcript_text=transcript_text or "")

    if not model:
        model = settings.AZURE_MODEL_CHAT_SUMMARY if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_CHAT_SUMMARY

    response = chat_completion(
        model=model,    
        temperature=0,
        messages=[{"role": "system", "content": system_prompt}, 
                  {"role": "user", "content": user_prompt}]       
        )
    
    log.info(f"Summary generated with model: {model}:")
    log.info("\n" + str(response.usage))   

    return response.choices[0].message.content


def format_summary_md( summary_text) -> str:
    new_text = summary_text.replace("1. ", "\n#### ", )
    new_text = new_text.replace("2. ", "\n#### ", )
    new_text = new_text.replace("3. ", "\n#### ", )
    new_text = new_text.replace("4. ", "\n#### ", )
    new_text = new_text.replace("5. ", "\n#### ", )
    new_text = new_text.replace("☐", "\n☐", )
    new_text = new_text.replace("ACTION_ENTRY_FORMAT:", "")
    return new_text



