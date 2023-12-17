# Use of an official Python runtime as a base image
FROM python:3.11.5

#Set up of working directory in the container i.e /app
WORKDIR /app

# Copy current directory contents into the container at /app
COPY . /app

# Install needed packages specified in requirements.txt recursively
RUN pip install -r requirements.txt

# Expose the port number the app runs on
EXPOSE 800

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "800"]