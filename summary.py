import argparse
from transformers import BartTokenizer, BartForConditionalGeneration

def generate_summary(prompt: str, model, tokenizer) -> str:
    """
    Generates text based on the given prompt using the BART model.

    Args:
        prompt (str): The prompt to guide the text generation.
        model: The BART model.
        tokenizer: The BART tokenizer.

    Returns:
        str: The generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True, padding='max_length')
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=150, 
        min_length=30, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def chunk_document(document: str, max_chunk_size: int) -> list:
    """
    Splits the document into chunks of specified maximum size.

    Args:
        document (str): The document to be split.
        max_chunk_size (int): The maximum size of each chunk.

    Returns:
        list: A list of text chunks.
    """
    chunks = [document[i:i + max_chunk_size] for i in range(0, len(document), max_chunk_size)]
    return chunks

def generate_combined_summary(chunks: list, model, tokenizer) -> str:
    """
    Generates a combined summary from multiple chunks.

    Args:
        chunks (list): A list of text chunks.
        model: The BART model.
        tokenizer: The BART tokenizer.

    Returns:
        str: The final combined summary.
    """
    # Generate summaries for each chunk
    chunk_summaries = [generate_summary(f"Summarize the following text in detail: {chunk}", model, tokenizer) for chunk in chunks]

    # Combine summaries and generate a final summary
    combined_summary_prompt = "Combine the following summaries into a final coherent summary: " + " ".join(chunk_summaries)
    final_summary = generate_summary(combined_summary_prompt, model, tokenizer)

    return final_summary

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate Abstract, Summary, and Conclusion using BART.")
    parser.add_argument("file", type=str, help="Path to the file containing the document to summarize.")
    args = parser.parse_args()

    # Read the document from the file
    try:
        with open(args.file, 'r', encoding='utf-8') as file:
            document = file.read()
    except FileNotFoundError:
        print(f"Error: The file {args.file} was not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not document.strip():
        print("Error: The document is empty.")
        return

    # Load the pre-trained BART model and tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Define chunk size and chunk the document
    max_chunk_size = 1000  # Adjust size based on your requirements
    chunks = chunk_document(document, max_chunk_size)

    # Generate and print each section
    print("Abstract:")
    abstract_prompt = f"Generate a concise abstract for the following text: {document}"
    print(generate_summary(abstract_prompt, model, tokenizer))
    print("\n")

    print("Summary:")
    print(generate_combined_summary(chunks, model, tokenizer))
    print("\n")

    print("Conclusion:")
    conclusion_prompt = f"Generate a concluding statement for the following text: {document}"
    print(generate_summary(conclusion_prompt, model, tokenizer))
    print("\n")

if __name__ == "__main__":
    main()

