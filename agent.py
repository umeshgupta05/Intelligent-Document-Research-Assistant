import os
from typing import TypedDict

from langchain_aws import ChatBedrockConverse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

NOT_FOUND = "NOT_FOUND"
MAX_TRIES = 2

REWRITE_SYSTEM = (
    "Rewrite the final question as a standalone search query, resolving pronouns and references "
    "using the conversation. Return only the query."
)
REWRITE_HUMAN = "Conversation:\n{history}\n\nQuestion: {question}"

GRADE_SYSTEM = "You grade whether a document excerpt is relevant to a question. Reply with only yes or no."
GRADE_HUMAN = "Question: {question}\n\nExcerpt:\n{excerpt}"

GENERATE_SYSTEM = (
    "Answer the question using only the context and the conversation. Be concise and accurate. "
    "If the context does not answer it, say that you could not find an answer."
)
GENERATE_HUMAN = "Conversation:\n{history}\n\nContext:\n{context}\n\nQuestion: {question}"

KNOW_SYSTEM = (
    "Answer from the conversation and your own general knowledge. "
    f"If you are not confident or the question needs current information, reply with exactly {NOT_FOUND}."
)
KNOW_HUMAN = "Conversation:\n{history}\n\nQuestion: {question}"

WEB_QUERY_SYSTEM = "Rewrite the question as a short keyword web search query. Return only the query."
WEB_QUERY_HUMAN = "Question: {question}"

GROUND_SYSTEM = "You check whether an answer is fully supported by the context. Reply with only yes or no."
GROUND_HUMAN = "Context:\n{context}\n\nAnswer:\n{answer}"

USEFUL_SYSTEM = "You check whether an answer addresses the question. Reply with only yes or no."
USEFUL_HUMAN = "Question: {question}\n\nAnswer:\n{answer}"


class State(TypedDict, total=False):
    question: str
    query: str
    history: str
    docs: list
    context: str
    refs: list
    answer: str
    source: str
    tries: int
    grounded: bool
    useful: bool


class Agent:
    def __init__(self, rag, memory):
        self.rag = rag
        self.memory = memory
        self.llm = ChatBedrockConverse(
            model=os.environ["BEDROCK_MODEL_ID"],
            region_name=os.getenv("AWS_REGION"),
            temperature=0,
        )
        self.web = TavilySearch(max_results=4)
        self.graph = self.build_graph()

    def ask(self, system, human, **variables):
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(variables).strip()

    def yes(self, system, human, **variables):
        return self.ask(system, human, **variables).lower().startswith("yes")

    def rewrite(self, state):
        if state["history"] == "None":
            return {"query": state["question"]}
        query = self.ask(REWRITE_SYSTEM, REWRITE_HUMAN, history=state["history"], question=state["question"])
        return {"query": query}

    def retrieve(self, state):
        return {"docs": self.rag.search(state["query"])}

    def grade(self, state):
        kept = []
        for doc in state["docs"]:
            if self.yes(GRADE_SYSTEM, GRADE_HUMAN, question=state["query"], excerpt=doc.page_content):
                kept.append(doc)
        context = "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in kept)
        refs = [f"{d.metadata['source']}: {d.page_content[:300]}" for d in kept]
        return {"context": context, "refs": refs, "source": "Documents", "tries": 0}

    def knowledge(self, state):
        answer = self.ask(KNOW_SYSTEM, KNOW_HUMAN, history=state["history"], question=state["question"])
        return {"answer": answer, "source": "Model knowledge", "refs": []}

    def search_web(self, state):
        query = self.ask(WEB_QUERY_SYSTEM, WEB_QUERY_HUMAN, question=state["query"])
        results = self.web.invoke({"query": query})["results"]
        context = "\n\n".join(f"{r['title']} ({r['url']})\n{r['content']}" for r in results)
        refs = [f"[{r['title']}]({r['url']})" for r in results]
        return {"context": context, "refs": refs, "source": "Web search", "tries": 0}

    def generate(self, state):
        answer = self.ask(
            GENERATE_SYSTEM,
            GENERATE_HUMAN,
            history=state["history"],
            context=state["context"],
            question=state["question"],
        )
        return {"answer": answer, "tries": state["tries"] + 1}

    def reflect(self, state):
        grounded = self.yes(GROUND_SYSTEM, GROUND_HUMAN, context=state["context"], answer=state["answer"])
        useful = self.yes(USEFUL_SYSTEM, USEFUL_HUMAN, question=state["question"], answer=state["answer"])
        return {"grounded": grounded, "useful": useful}

    def after_rewrite(self, state):
        if self.rag.chunks:
            return "retrieve"
        return "knowledge"

    def after_grade(self, state):
        if state["context"]:
            return "generate"
        return "knowledge"

    def after_knowledge(self, state):
        if NOT_FOUND in state["answer"]:
            return "search_web"
        return END

    def after_reflect(self, state):
        if state["grounded"] and state["useful"]:
            return END
        if state["tries"] >= MAX_TRIES:
            return END
        if not state["grounded"]:
            return "generate"
        if state["source"] == "Documents":
            return "search_web"
        return END

    def build_graph(self):
        graph = StateGraph(State)
        graph.add_node("rewrite", self.rewrite)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("grade", self.grade)
        graph.add_node("knowledge", self.knowledge)
        graph.add_node("search_web", self.search_web)
        graph.add_node("generate", self.generate)
        graph.add_node("reflect", self.reflect)
        graph.add_edge(START, "rewrite")
        graph.add_conditional_edges("rewrite", self.after_rewrite, ["retrieve", "knowledge"])
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges("grade", self.after_grade, ["generate", "knowledge"])
        graph.add_conditional_edges("knowledge", self.after_knowledge, ["search_web", END])
        graph.add_edge("search_web", "generate")
        graph.add_edge("generate", "reflect")
        graph.add_conditional_edges("reflect", self.after_reflect, ["generate", "search_web", END])
        return graph.compile()

    def run(self, question):
        start = {"question": question, "history": self.memory.as_text() or "None"}
        state = {}
        steps = []
        for update in self.graph.stream(start, stream_mode="updates"):
            for node, values in update.items():
                steps.append(node)
                state.update(values)
        return {
            "answer": state["answer"],
            "source": state["source"],
            "references": state["refs"],
            "steps": steps,
        }
