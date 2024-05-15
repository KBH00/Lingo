from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def extract_medical_terms(text):
    """
    Extract medical terms from a given text block, assuming each term is listed in a numbered format.

    Args:
    - text (str): A block of text containing medical terms listed in a numbered format.
    
    Returns:
    - list: A list of extracted medical terms.
    """
    extracted_terms = []
    lines = text.split('\n')
    
    for line in lines:
        if line.strip().startswith(tuple(str(i) for i in range(1, 10))): 
            term = line.split('. ', 1)[-1]
            term = term.split('(', 1)[0].strip()
            extracted_terms.append(term)
    
    return extracted_terms


template = """Question: {question}
Context: {context}

Answer: Let's consider the following context to find the best synonyms in the medical domain for the term:
"""

prompt_template = PromptTemplate(template=template, input_variables=["question", "context"])
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 30
n_batch = 2000
llm = LlamaCpp(
    model_path="C:/Users/kbh/Code/project2/llm/models/llama-2-7b-chat.Q2_K.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    temperature=0,
    callback_manager=callback_manager,
    verbose=False,
    n_ctx=1000
)

llm_chain = LLMChain(prompt=prompt_template, llm=llm)
word = "CT"
context = "In scenarios where quick imaging is needed to diagnose internal injuries, a rapid method is often required."
question = f"Recommend 5 synonyms of {word} in the medical domain, given the context."

inputs = {
    "question": question,
    "context": context
}

response = llm_chain.invoke(inputs)
print(extract_medical_terms(response['text']))

