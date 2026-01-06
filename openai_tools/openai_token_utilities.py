
import tiktoken

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""

    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"]:
        model = "gpt-3.5-turbo-0301"
    elif model in ["gpt-4", "gpt-4-32k"]:
        model = "gpt-4-0314"
    elif model in ["gpt-4-1106-preview", "gpt-4-turbo-preview", "gpt-4-0125-preview", "gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o", "gpt-4o-mini", "o3-mini"]:
        model = "gpt-4-0314"
    else:
        model = "gpt-3.5-turbo-0301"

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-3.5-turbo-16k":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-3.5-turbo-1106":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-4-32k":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            value_text = str(value)
            num_tokens += len(encoding.encode(value_text))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    
    return num_tokens



def num_tokens_from_text(text, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"]:
        model = "gpt-3.5-turbo-0301"
    elif model in ["gpt-4", "gpt-4-32k"]:
        model = "gpt-4-0314"
    elif model in ["gpt-4-1106-preview", "gpt-4-turbo-preview", "gpt-4-0125-preview", "gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o", "gpt-4o-mini", "o3-mini"]:
        model = "gpt-4-0314"
    else:
        model = "gpt-3.5-turbo-0301"
 
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model == "gpt-3.5-turbo-16k":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_text(text, model="gpt-3.5-turbo-0301")
    elif model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_text(text, model="gpt-3.5-turbo-0301")
    elif model == "gpt-3.5-turbo-1106":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(text, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_text(text, model="gpt-4-0314")
    elif model == "gpt-4-32k":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_text(text, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_token_from_text() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    # num_tokens = 0
    # for message in messages:
    #     num_tokens += tokens_per_message
    #     for key, value in message.items():
    #         num_tokens += len(encoding.encode(value))
    #         if key == "name":
    #             num_tokens += tokens_per_name
    # num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    
    return len(encoding.encode(text))


def num_tokens_from_file(filename, model="gpt-3.5-turbo-0301"):
    with open(filename, "r", encoding="utf-8") as f:
        text=f.read()
    return num_tokens_from_text(text, model)


 

# def cost_for_tokens(model: str, text=None, messages=None):
#     return cost_for_tokens_in(model=model, text=text, messages=messages)

def cost_for_tokens_in(model: str, text=None, messages=None):
    if model == 'gpt-3.5-turbo':
        cost = 0.001 / 1000
    elif model == 'gpt-3.5-turbo-16k':
        cost = 0.001 / 1000
    elif model == 'gpt-3.5-turbo-1106':
        cost = 0.001 / 1000    
    elif model == 'gpt-3.5-turbo-0125':
        cost = 0.5 / 1000000
    elif model == 'gpt-4':
        cost = 30 / 1000000
    elif model == 'gpt-4-32k':
        cost = 60 / 1000000
    elif model == 'gpt-4-1106-preview':
        cost = 10 / 1000000
    elif model == 'gpt-4-0125-preview':
        cost = 10 / 1000000    
    elif model == 'gpt-4-turbo-2024-04-09':
        cost = 10 / 1000000    
    elif model == 'gpt-4o-2024-05-13':
        cost = 5 / 1000000
    elif model == 'gpt-4o-2024-08-06':
        cost = 2.5 / 1000000
    elif model == 'gpt-4o-mini':
        cost = 0.15 / 1000000        
    elif model == 'o3-mini':
        cost = 1.1 / 1000000        
    elif model == 'gpt-4-turbo-preview':
        cost = 10 / 1000000
    elif model == 'text-embedding-ada-002':
        cost = 0.1 / 1000000
    elif model == 'text-embedding-3-small':
        cost = 0.02 / 1000000
    elif model == 'text-embedding-3-large':
        cost = 0.13 / 1000000
    else:
        cost = 0.003 / 1000
 
    if text is not None:
        return num_tokens_from_text(text) * cost 
    if messages is not None:
        return num_tokens_from_messages(messages) * cost 
    return cost

def cost_for_tokens_out(model: str, text=None):
    if model == 'gpt-3.5-turbo':
        cost = 0.002 / 1000
    elif model == 'gpt-3.5-turbo-16k':
        cost = 0.002 / 1000
    elif model == 'gpt-3.5-turbo-1106':
        cost = 0.002 / 1000
    elif model == 'gpt-3.5-turbo-0125':
        cost =1.5 / 1000000
    elif model == 'gpt-4':
        cost = 60 / 1000000
    elif model == 'gpt-4-32k':
        cost = 120 / 1000000
    elif model == 'gpt-4-1106-preview':
        cost = 30 / 1000000
    elif model == 'gpt-4-0125-preview':
        cost = 30 / 1000000
    elif model == 'gpt-4-turbo-2024-04-09':
        cost = 30 / 1000000
    elif model == 'gpt-4o-2024-05-13':
        cost = 15 / 1000000
    elif model == 'gpt-4o-2024-08-06':
        cost = 10 / 1000000
    elif model == 'gpt-4o-mini':
        cost = 0.6 / 1000000     
    elif model == 'o3-mini':
        cost = 4.4 / 1000000     
    elif model == 'gpt-4-turbo-preview':
        cost = 30 / 1000000
    else:
        cost = 0.006 / 1000

    if text is not None:
        return num_tokens_from_text(text) * cost 
    return cost 


def max_tokens_for_model(model: str):
    if model == 'gpt-3.5-turbo':
        max_tokens = 4096
    elif model == 'gpt-3.5-turbo-16k':
        max_tokens = 16385
    elif model == 'gpt-3.5-turbo-1106':
        max_tokens = 16385
    elif model == 'gpt-3.5-turbo-0125':
        max_tokens = 16385
    elif model == 'gpt-4':
        max_tokens = 8192
    elif model == 'gpt-4-32k':
        max_tokens = 32768
    elif model == 'gpt-4-1106-preview':
        max_tokens = 128000
    elif model == 'gpt-4-0125-preview':
        max_tokens = 128000
    elif model == 'gpt-4-turbo-2024-04-09':
        max_tokens = 128000
    elif model == 'gpt-4o-2024-05-13':
        max_tokens = 128000    
    elif model == 'gpt-4o-2024-08-06':
        max_tokens = 128000
    elif model == 'gpt-4o-mini':
        max_tokens = 128000   
    elif model == 'o3-mini':
        max_tokens = 200000   
    elif model == 'text-embedding-ada-002':
        max_tokens = 8192
    else:
        max_tokens = 16384
    return max_tokens



def cost_for_tokens_in_messages(messages, model):
    if len(messages)==0:
        return 0
    if messages[len(messages)-1]["role"] == "assistant":
        num_of_tokens_in = num_tokens_from_messages(messages=messages[:-1], model=model) 
        num_of_tokens_out = num_tokens_from_messages(messages=messages[len(messages)-1:], model=model)
    else:
        num_of_tokens_in = num_tokens_from_messages(messages=messages, model=model) 
        num_of_tokens_out = 0
    
    cost = cost_for_tokens_in(model=model)*num_of_tokens_in + cost_for_tokens_out(model=model)*num_of_tokens_out
    return cost
    

def cost_of_tokens_by_usage(usage, model):
    return cost_for_tokens_in(model)*usage.prompt_tokens + cost_for_tokens_out(model)*usage.completion_tokens   
