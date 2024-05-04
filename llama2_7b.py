from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class llama2_7b():

    def __init__(self, context, cntxt_len, targetWord, model) -> None:
        self.context = context
        self.cntxt_len = cntxt_len
        self.word = targetWord
        self.model = model

    def inference(self) -> str:            
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

        #callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        llm = self.model
        llm_chain = LLMChain(prompt=prompt_template, llm=llm)
        response = llm_chain.invoke(prompt)

        return response['text']