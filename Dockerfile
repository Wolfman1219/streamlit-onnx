# Use the official Debian 11 image
FROM debian:11

# Set environment variables
ENV PYTHON_VERSION=3.9

# Install necessary packages
RUN apt-get update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv && \
    apt-get clean

# Create and set the working directory
WORKDIR /app

# Copy the project files to the container
COPY . /app/

# Create a virtual environment and install dependencies
RUN python${PYTHON_VERSION} -m venv venv && \
    . venv/bin/activate && \
    cat requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip install
# pip install --no-cache-dir -r requirements.txt

# Expose the necessary port
EXPOSE 8501

# Command to run your application
CMD ["venv/bin/streamlit", "run", "custom_yolov8.py", "--server.port", "8501"]
