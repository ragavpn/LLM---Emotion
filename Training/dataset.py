import pandas as pd
import jsonlines
import os

def format_text(input_text, emotion1, dialogue_emotion1, output_text, emotion2, dialogue_emotion2):
    # Generate an empathetic response
    response = (
        f'"{input_text}" - this shows {emotion1} and from past conversation, it also shows {dialogue_emotion1}. '
        f'Due to these emotions being present, an empathetic response is crucial to convey understanding and support. '
        f'Therefore, the two necessary empathetic emotions in a response would be {emotion2} and {dialogue_emotion2}. '
        f'The response to the above statement can be "{output_text}".'
    )
    return response


# Read the CSV file
csv_file = "train.csv"  
df = pd.read_csv(csv_file)

# Initialize variables to store conversation context
input_text = ""
emotion1 = ""
dialogue_emotion1 = ""

# Provide some example data for fine-tuning
training_data = []

# Loop through the CSV rows
for index, row in df.iterrows():
    if index % 2 == 0:
        # This row contains "input" and related emotions
        input_text = row["text"]
        emotion1 = row["emotion"]
        dialogue_emotion1 = row["dialogue_emotion"]
    else:
        # This row contains "output" and related emotions
        output_text = row["text"]
        emotion2 = row["emotion"]
        dialogue_emotion2 = row["dialogue_emotion"]
        # Generate an empathetic response
        response = format_text(input_text, emotion1, dialogue_emotion1, output_text, emotion2, dialogue_emotion2)
        training_data.append(response)

csv_file = "test.csv"  
df = pd.read_csv(csv_file)
val_data = []

# Loop through the CSV rows
for index, row in df.iterrows():
    if index % 2 == 0:
        # This row contains "input" and related emotions
        input_text = row["text"]
        emotion1 = row["emotion"]
        dialogue_emotion1 = row["dialogue_emotion"]
    else:
        # This row contains "output" and related emotions
        output_text = row["text"]
        emotion2 = row["emotion"]
        dialogue_emotion2 = row["dialogue_emotion"]
    
        # Generate an empathetic response
        response = format_text(input_text, emotion1, dialogue_emotion1, output_text, emotion2, dialogue_emotion2)
        val_data.append(response)

training_data = training_data[:3000]
val_data = val_data[:1000]


def write_to_jsonl(data, file_path):
    # Clear the file before writing
    if os.path.exists(file_path):
        os.remove(file_path)
    with jsonlines.open(file_path, mode='a') as writer:
        for item in data:
            writer.write({'text': item})

file_path_train = 'notes.jsonl'
file_path = 'notes_val.jsonl'
write_to_jsonl(training_data, file_path_train)
write_to_jsonl(val_data, file_path)

