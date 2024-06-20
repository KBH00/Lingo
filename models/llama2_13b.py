from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class llama2_13b():
    
    def __init__(self) -> None:
        
        self.model = LlamaCpp(
        model_path="./models/checkpoint/"+"llama-2-13b-chat.Q3_K_M.gguf",
        n_gpu_layers=50,
        n_batch=2000,
        temperature=0,
        #callback_manager=CallbackManager([]),
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
        n_ctx=1800,
        )        

    def extract_medical_terms(self, text) -> list:
        """
        Extract medical terms from a given text block, assuming each term is listed in a numbered format.

        Args:
        - text (str): A block of text containing medical terms listed in a numbered format.

        Returns:
        - list: A list of extracted medical terms.
        """

        extracted_terms = []
        lines = text.split("\n")
        for line in lines:
            if line.strip().startswith(tuple(str(i) for i in range(1, 6))):
                term = line.split(". ", 1)[-1]
                if "(" in term:
                    term = term.split("(", 1)[0]
                extracted_terms.append(term)
                
        return extracted_terms
    
    def suggestion(self, targetWord, context, cntxt_len) -> list:
        self.context = context
        self.cntxt_len = cntxt_len
        self.word = targetWord
                    
        template = ["""Question: {question}
        
        Answer: Do not say anything else but answer""",

        """Question: {question}
        Context: {context}

        Answer: Do not say anything else but answer"""
        ]

        #self.context = "In scenarios where quick imaging is needed to diagnose internal injuries, a rapid method is often required."
        question = f"Recommend 5 synonyms of {self.word} in the medical domain, consider context."

        input_context = {
            "question": question,
            "context": self.context
        }
        
        input_without_context ={
            f"""
            Question: recommend 5 synonyms of {self.word} in the medical domain without definition of words. Do not say anything else.
            """
        }
        prompt = [input_without_context, input_context]

        if self.cntxt_len == 0:
            template = template[0]
            prompt = prompt[0]
        else:
            template = template[1]
            prompt = prompt[1]

        prompt_template = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = prompt_template | self.model
        response = llm_chain.invoke(prompt)

        return self.extract_medical_terms(response)
