# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make a directory to mount the NVMe drive
RUN mkdir /nvme_drive

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME mbros1

# Set up NVIDIA runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Run app.py when the container launches
CMD ["python", "app.py"]
