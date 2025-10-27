def _default_compute_metrics(self, eval_pred):
    """Compute evaluation metrics for generation."""
    predictions, labels = eval_pred
    
    # Handle predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Convert to numpy array and ensure correct shape
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # For seq2seq models, predictions should be 2D: (batch_size, seq_length)
    # If predictions is 3D, take argmax along last dimension
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)
    
    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
    
    # Convert to list of lists for batch_decode (each element should be a list of ints)
    predictions = predictions.tolist()
    labels = labels.tolist()
    
    try:
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        print(f"Error during decoding: {e}")
        print(f"Predictions shape: {np.array(predictions).shape}")
        print(f"Labels shape: {np.array(labels).shape}")
        print(f"Sample prediction type: {type(predictions[0]) if predictions else 'empty'}")
        raise
    
    # Clean whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Compute metrics
    metrics = {}
    
    # ROUGE scores
    try:
        rouge = load('rouge')
        rouge_results = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        metrics.update({
            'rouge1': rouge_results['rouge1'],
            'rouge2': rouge_results['rouge2'],
            'rougeL': rouge_results['rougeL']
        })
    except Exception as e:
        print(f"Warning: Could not compute ROUGE: {e}")
    
    # BLEU score
    try:
        bleu = load('bleu')
        bleu_results = bleu.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )
        metrics['bleu'] = bleu_results['bleu']
    except Exception as e:
        print(f"Warning: Could not compute BLEU: {e}")
    
    # Average generation length
    metrics['gen_len'] = np.mean([len(pred.split()) for pred in decoded_preds])
    
    return metrics
