FROM pytorch/pytorch:latest

COPY . /workspace
WORKDIR /workspace

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "app.py", "-t", "query_restful"]

LABEL author="madjakarta@gmail.com"
LABEL type="app"
LABEL kind="QA"
LABEL avatar="None"
LABEL description="LegalQA for Korean"
LABEL documentation="https://github.com/haven-jeon/LegalQA.git"
LABEL keywords="[NLP, text, QA, KoBERT]"
LABEL license="apache-2.0"
LABEL name="LegalQA"
LABEL platform="linux/amd64"
LABEL update="None"
LABEL url="https://github.com/haven-jeon/LegalQA.git"
LABEL vendor="gogamza"
LABEL version="0.0.1"
