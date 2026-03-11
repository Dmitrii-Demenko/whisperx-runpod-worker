FROM dimademenko/whisperx-base:latest

# Copy handler source
COPY rp_handler.py /rp_handler.py
COPY rp_schema.py /rp_schema.py

CMD ["python3", "-u", "/rp_handler.py"]
