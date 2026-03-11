import json

def build_global_vocabulary(json_filepaths):
    """
    Scrapes a list of Diplomacy JSONL game files to build a master vocabulary
    of all unique orders ever issued.
    """
    global_vocab = set()
    
    # We always need a dummy action for provinces without units
    global_vocab.add("DUMMY") 
    
    for filepath in json_filepaths:
        with open("datasets/"+filepath, 'r', encoding='utf-8') as f:
            # Iterate through the file line by line
            for line in f:
                line = line.strip()
                if not line:
                    continue # Skip blank lines
                
                # Parse the single line as a JSON object
                game = json.loads(line)
                
                # Process the game data
                for phase in game.get('phases', []):
                    orders_dict = phase.get('orders', {})
                    
                    # Orders are stored by power: {"FRANCE": ["A PAR - BUR", ...]}
                    for power, order_list in orders_dict.items():
                        
                        # Fallback to an empty list if order_list is None
                        order_list = order_list or [] 
                        
                        for order in order_list:
                            # Add the exact string to our unique set
                            global_vocab.add(order)
                            
    # Sort it so the indices are deterministic every time you load it
    sorted_vocab = sorted(list(global_vocab))
    
    # Map strings to integer IDs
    vocab_to_idx = {order: idx for idx, order in enumerate(sorted_vocab)}
    idx_to_vocab = {idx: order for idx, order in enumerate(sorted_vocab)}
    
    print(f"Total unique orders found: {len(vocab_to_idx)}")
    return vocab_to_idx, idx_to_vocab

# --- Example Usage ---
filepaths = ["standard_no_press.jsonl", "standard_press_with_msgs.jsonl", "standard_press_without_msgs.jsonl", "standard_public_press.jsonl"] 
vocab_to_idx, idx_to_vocab = build_global_vocabulary(filepaths)

# Save this to a file so you don't have to rebuild it every time you train!
with open('diplomacy_vocab.json', 'w') as f:
    json.dump({'vocab_to_idx': vocab_to_idx}, f)