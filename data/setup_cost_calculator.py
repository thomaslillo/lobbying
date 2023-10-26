import pandas as pd
import tiktoken

# Load the CSV data into a Pandas DataFrame
file_path = 'Data/VariableMetadata2022.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Define a function to count tokens in a given text
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Apply the count_tokens function to the "varname" column and create a new column "tokens_count"
df['tokens_count'] = df['VARIABLEDESC'].apply(num_tokens_from_string)

# Now, df['tokens_count'] contains the number of tokens in the "varname" column
total_tokens = df['tokens_count'].sum()
total_cost = (total_tokens / 1000) * 0.0001

# You can print the DataFrame to see the results
print("The Total Tokens: ")
print(str(total_tokens))
print("\n The Total Cost: ")
print(str(total_cost))
