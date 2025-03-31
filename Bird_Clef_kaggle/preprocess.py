import torch
import torchaudio
import threading
import queue
import time
import concurrent.futures
import pickle
from pathlib import Path
import librosa
import gc
import numpy as np
import matplotlib.pyplot as plt

class Preprocessor:
    def __init__(self):
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # Load the Silero VAD model onto GPU
        torch.set_num_threads(4)  # Avoid unnecessary CPU usage
        self.vad_model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, trust_repo=True)
        self.vad_model.to(self.device)  # Move model to GPU
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils

    def RemoveHumanVoice(self, waveform, sr):

        waveform = waveform.to(self.device)
        # Apply VAD
        with torch.no_grad():
            speechTimestamps = self.get_speech_timestamps(waveform, self.vad_model, sampling_rate=sr)
        
        speechSegments = np.zeros(waveform.shape[0], dtype = bool)
        for timeStamp in speechTimestamps:
            speechSegments[timeStamp['start']:timeStamp['end']] = True
        return waveform.cpu().t().numpy()[~speechSegments]


# Thread-safe queue for processing
data_queue = queue.Queue(maxsize=4)

# Condition variable for synchronization
condition = threading.Condition()

stopEvent = threading.Event()

def LoadData(item):
    """Load data and add it to the queue when space is available."""
    try:
        waveform, sr = librosa.load(str(item), sr=None)  # Load audio file with librosa
        # Check if the waveform has more than one channel (e.g., stereo)
        if waveform.ndim > 1:  # This checks for multi-dimensional arrays (e.g., stereo audio)
            waveform = np.mean(waveform, axis=0, keepdims=True)  # Average over channels to convert to mono
        data = {"fileName": item, "waveform": torch.tensor(waveform), "samplingRate": sr}
        return data
    except Exception as e:
        print(f"Error loading {item}: {e}")
        return None

# Producer thread function
def Producer(files_list):
    """Load data in multiple threads and add it to the queue."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # Use 5 threads for parallel loading
        for file_id in files_list:
            future = executor.submit(LoadData, file_id)
            data = future.result() # get result immediately
            del future
            
            # Check if the future completed successfully
            if data is not None:
                print(f"Produced data: {data['fileName']}")  # Debugging print
                
                # Synchronize the addition to the queue
                with condition:
                    # Wait if the queue is full
                    while data_queue.full():
                        print(f"Queue is full, waiting for space... (size: {data_queue.qsize()})")
                        condition.wait()  # Wait for space in the queue
                    
                    # Add data to the queue
                    data_queue.put(data)
                    print(f"Data added to queue: {data['fileName']}, Queue size: {data_queue.qsize()}")
                    
                    # Notify the consumer thread that data is available
                    condition.notify()  # Notify consumer after adding data

                del data # delete data explicitly as it can be large

            else:
                print(f"Skipping file due to loading error: {future}")

    # Set stop event once all data has been added
    stopEvent.set()
    print("Producer finished processing.")

# Consumer thread function
def Consumer(save_file):
    """Consume data from the queue, process it, and dump to pickle."""
    preprocessData = Preprocessor()
    while True:
        with condition:
            # Wait until data is available or stop event is set
            while data_queue.empty() and not stopEvent.is_set():
                print("Consumer waiting for data...")
                condition.wait()  # Release lock while waiting for data
            
            # Exit condition: stop event is set and queue is empty
            if stopEvent.is_set() and data_queue.empty():
                print("Consumer has finished processing all data.")
                break

            # Process data from the queue
            item = data_queue.get()
            try:
                voiceRemoved = preprocessData.RemoveHumanVoice(item["waveform"], item["samplingRate"])
                result = {'fileName': str(item["fileName"]), 'segmentation': voiceRemoved, 'Sampling_rate': item["samplingRate"]}
                print(f"Consuming: {item['fileName']}")
                                
                # Save to pickle file
                with open(save_file, "ab") as f:
                    pickle.dump(result, f)

            except Exception as e:
                print(f"[ERROR] In inference {e}")

            data_queue.task_done()  # Mark the task as done

            # Notify producer that space is available in the queue
            condition.notify()  # Notify producer that the queue is not full anymore

    print("Consumer thread finished.")

# Main execution
if __name__ == "__main__":
    params = dict()
    params["input_folder"] = r"C:\Users\vivek\Downloads\birdclef-2025\train_audio"
    params["saveFile"] = r"C:\Users\vivek\Downloads\birdclef-2025\train_audio_humanSpeechRemoved.pkl"

    # Delete existing file (ensures fresh start)
    Path(params["saveFile"]).unlink(missing_ok=True)

    # Get a list of audio files in the folder
    inputAudioFiles = list(Path(params["input_folder"]).rglob("*.ogg"))
    
    if not inputAudioFiles:
        print("No .ogg files found in the specified directory.")
    else:
        print(f"Found {len(inputAudioFiles)} files.")
    
        # Start the producer and consumer threads
        producer_thread = threading.Thread(target=Producer, args=(inputAudioFiles,))
        producer_thread.start()
        
        consumer_thread = threading.Thread(target=Consumer, args=(params["saveFile"],))
        consumer_thread.start()
        
        # Wait for both threads to finish
        producer_thread.join()
        consumer_thread.join()

    print("All processing complete.")
