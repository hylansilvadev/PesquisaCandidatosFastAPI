from fastapi import FastAPI
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os

load_dotenv()


app = FastAPI()

# Carregar as chaves da API
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")


# Definir o modelo de entrada
class JobRequirements(BaseModel):
    job_requirements: str


# Criar o agente pesquisador
search_tool = SerperDevTool()
researcher = Agent(
    role='Recrutador Senior',
    goal='Encontrar os melhores perfis para trabalhar baseados nos requisitos da vaga',
    verbose=True,
    memory=True,
    model='gpt-4o-mini',
    backstory=(
        "Experiencia na area de dados e formação academica em Recursos Humanos e "
        "Especilista em Linkedin, tem dominio das principais taticas de busca de profissionais"
    ),
    tools=[search_tool],
    iteration_limit=50,  # Ajuste o limite de iterações conforme necessário
    # Ajuste o limite de tempo (em segundos) conforme necessário
    time_limit=600
)


# Definir a rota para executar a tarefa


@app.post("/research_candidates")
async def research_candidates(req: JobRequirements):
    # Criar a tarefa de pesquisa
    research_task = Task(
        description=(
            f"Pesquise candidatos potenciais para o cargo de {
                req.job_requirements}. "
            f"Reúna uma lista de 5 candidatos que atendam aos requisitos."
        ),
        expected_output="""Uma lista com 5 candidatos potenciais, incluindo informações de contato, 
                       breve descrição do perfil e a URL para o perfil.""",
        tools=[search_tool],
        agent=researcher,
    )

    # Criar a equipe e executar a tarefa
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential
    )

    result = crew.kickoff(inputs={'job_requirements': req.job_requirements})
    return {"result": result}

# Rodar o servidor usando Uvicorn
if __name__ == "__main__":
    import uvicorn
    print(">>>>>>>>>>>> version V0.0.1")
    uvicorn.run(app, host="0.0.0.0", port=8000)
