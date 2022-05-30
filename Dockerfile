
FROM python:3.8.8

WORKDIR /pd_rta

COPY requirements.txt .

COPY RTA_Model_pickle.pkl .

ADD templates ./templates

COPY model.py .

COPY app.py .

RUN pip install -r requirements.txt

CMD ["python", "./app.py"]
