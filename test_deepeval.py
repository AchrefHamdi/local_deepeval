from deepeval.tracing import trace, TraceType
from openai import OpenAI
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric
import deepeval
 
import os
print(deepeval.tracing.__file__)
os.environ["OPENAI_API_KEY"] = ""
 
client = OpenAI()
 
class Chatbot:
    def __init__(self):
        self.completion_tokens=None
        self.prompt_tokens=None
        self.total_tokens=None
        self.model = None
        self.chatId = None
        self.retrieval_nodes=None

        pass
 
    @trace(type=TraceType.LLM, name="OpenAI", model="gpt-4")
    def llm(self, input):
        response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {"role": "user", "content": input},
                    ],
                   
                )
        #token_cost = response.get_cost()
        #token_usage = response["total_tokens"]
        self.completion_tokens = str(response.usage.completion_tokens)
        self.prompt_tokens = str(response.usage.prompt_tokens)
        self.total_tokens = str(response.usage.total_tokens)
        self.model = str(response.model)
        self.chatId = str(response.id)
        print("*********************",response.usage.prompt_tokens)
 
        return response.choices[0].message.content
 
    @trace(type=TraceType.EMBEDDING, name="Embedding", model="text-embedding-ada-002")
    def get_embedding(self, input):
        response = client.embeddings.create(
            input=input,
            model="text-embedding-ada-002"
        )
        #print("this is response",response)
        embedding = response.data[0].embedding
        return embedding
 
    @trace(type=TraceType.RETRIEVER, name="Retriever")
    def retriever(self, input=input):
        embedding = self.get_embedding(input)
 
        # Replace this with an actual vector search that uses embedding
        list_of_retrieved_nodes = ["Retrieval Node 1", "Retrieval Node 2"]
        self.retrieval_nodes=str(list_of_retrieved_nodes)
        return list_of_retrieved_nodes
 
    @trace(type=TraceType.TOOL, name="Search")
    def search(self, input):
        # Replace this with an actual function that searches the web
        title_of_the_top_search_results = "Search Result: " + input
        return title_of_the_top_search_results
 
 
    @trace(type=TraceType.TOOL, name="Format")
    def format(self, retrieval_nodes, input):
        prompt = "You are a helpful assistant, based on the following information: \n"
        for node in retrieval_nodes:
            prompt += node + "\n"
        prompt += "Generate an unbiased response for " + input + "."
        return prompt
 
    @trace(type=TraceType.AGENT, name="Chatbot")
    def query(self, user_input=input):
        top_result_title = self.search(user_input)
        retrieval_results = self.retriever(top_result_title)
        prompt = self.format(retrieval_results, top_result_title)
        return self.llm(prompt)

    
 
 
chatbot = Chatbot()


#1- run without chatbot as argument for deepeval
#2- add chatbot as argument & save before run flask
def test_hallucination():
    context = [ "Paul Graham is a computer scientist, entrepreneur, venture capitalist, and author." ]
 
    input = "Who is Paul Graham?"
    results = chatbot.llm(input)
    #print("################", chatbot.completion_tokens)
   
 
    metric = HallucinationMetric(threshold=0.8)
    test_case = LLMTestCase(
        input=input,
        actual_output=chatbot.query(user_input=input),
        # token_cost = chatbot.token_cost,
        # token_usage = chatbot.total_tokens,
        context=context,
    )
 
    # return {
       
    #     "model":chatbot.model,
    #     "input":input,
    #     "output":test_case.actual_output,
    #     "conversation_id":chatbot.chatId,
    #     "completion_token" : chatbot.completion_tokens,
    #     "prompt_token" : chatbot.prompt_tokens,
    #     "token_usage" : chatbot.total_tokens,
    #     # "Embedding":chatbot.get_embedding(input),
    #     "retriever":chatbot.retriever(input),
    #     "search":chatbot.search(input),
    #     "format":chatbot.format(input,chatbot.retrieval_nodes),
    #     "query":chatbot.query(input)
    # }
    # At the end of your LLM call
    
    deepeval.track(
        event_name="",
        model=chatbot.model,
        input=input,
        output=test_case.actual_output,
        distinct_id="",
        conversation_id=chatbot.chatId,
        retrieval_context=["..."],
        completion_time=8.23,
        #completion_token = chatbot.completion_tokens,
        #prompt_token = chatbot.prompt_tokens,
        token_usage = chatbot.total_tokens,
        additional_data={"example": "example"},
        fail_silently=True,
       
        run_on_background_thread=True
    )
    #print("------------",deepeval.track)
    print("done !")
    assert_test(test_case, [metric])

 
 
