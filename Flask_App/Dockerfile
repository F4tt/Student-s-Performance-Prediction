FROM python:3.12.3
WORKDIR /app
ADD . /app
RUN pip install --trusted-host pypi.python.org --progress-bar off -r requirements.txt
EXPOSE 5000
ENV NAME OpentoALL
CMD ["python","app.py"]