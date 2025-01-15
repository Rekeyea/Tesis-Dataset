from kafka.consumer import KafkaConsumer
from kafka.errors import KafkaError
import json
from datetime import datetime
import signal
import sys

class PatientDataConsumer:
    def __init__(self):
        self.consumer = None
        self.running = True
        
        # Set up signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """Handle shutdown gracefully"""
        print("\nShutting down consumer...")
        self.running = False

    def setup_consumer(self):
        """Initialize the Kafka consumer"""
        try:
            self.consumer = KafkaConsumer(
                'patient_measurements',
                bootstrap_servers=['localhost:9091', 'localhost:9092', 'localhost:9093'],
                group_id='patient_data_consumer_group',
                auto_offset_reset='earliest',  # Start from earliest message if no offset is stored
                enable_auto_commit=True,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                consumer_timeout_ms=1000  # Time out after 1 second of no messages
            )
        except KafkaError as e:
            print(f"Error setting up Kafka consumer: {e}")
            sys.exit(1)

    def process_message(self, message):
        """Process each message from Kafka"""
        try:
            # Extract data from message
            data = message.value
            
            # Convert ISO format strings back to datetime for processing if needed
            timestamp = datetime.fromisoformat(data['timestamp'])
            
            # Print formatted message
            print("\nReceived Patient Record:")
            print(f"Device ID: {data['device_id']}")
            print(f"Measurement Type: {data['measurement_type']}")
            print(f"Raw Value: {data['raw_value']}")
            print(f"Timestamp: {timestamp}")
            print("-" * 50)
            
            # Here you could add additional processing, storage, or analysis of the data
            
        except Exception as e:
            print(f"Error processing message: {e}")
            print(f"Problematic message: {message.value}")

    def run(self):
        """Main loop to consume messages"""
        self.setup_consumer()
        
        print("Starting to consume messages... Press Ctrl+C to exit")
        
        try:
            while self.running:
                try:
                    # Consume messages in a loop
                    for message in self.consumer:
                        if not self.running:
                            break
                        self.process_message(message)
                except KafkaError as e:
                    print(f"Error consuming messages: {e}")
                    continue
                
        finally:
            # Clean up
            if self.consumer:
                try:
                    self.consumer.close()
                    print("\nConsumer closed successfully")
                except Exception as e:
                    print(f"Error closing consumer: {e}")

def main():
    consumer = PatientDataConsumer()
    consumer.run()

if __name__ == "__main__":
    main()