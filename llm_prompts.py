class LLMPrompts:
    """
    Static prompts for LLM interactions in the trust framework
    """

    # Prompt for decomposing text into smaller units
    DECOMPOSE = (
        "Break down the following text into several self-contained information units.\n"
        "Each unit should:\n"
        "1. Be a complete, standalone piece of information\n"
        "2. Retain its original meaning without the surrounding context\n"
        "3. Should not use pronoun to refer to the information in other units\n"
        "4. There can be some overlap between units when necessary\n"
        "5. Be factually accurate and preserve the original intent\n\n"
        "Text: {text}\n\n"
        "If the text is short(basic) enough and cannot be decomposed, just return 'False', "
        "else return a JSON list of decomposed text units, one per line."
    )
    # prompt for reasoning about if text_1 implied text_2
    IMPLICATION_CHECK = (
        "Check if the following text_1 implies the following text_2:\n\n"
        "Text_1: {text_1}\n"
        "Text_2: {text_2}\n\n"
        "You check if text_1 implies text_2 purely based on the information in text_1 and text_2, and not any other prior information.\n"
        "Return Positive if based on text_1 you can conclude every statements intext_2 are true\n"
        "Return Negative if based on text_1 you can conclude at least one statement in text_2 is false\n"
        "Return Not Sure if based on text_1 you cannot conclude text_2 is true or false\n"
    )

    # Prompt for trust score calculation
    TRUST_EVALUATION = (
        "Evaluate the trustworthiness of the following information:\n\n"
        "Text: {text}\n\n"
        "Consider:\n"
        "1. Factual precision and specificity\n"
        "2. Source reliability indicators\n"
        "3. Internal consistency\n"
        "4. Verifiability of claims\n"
        "5. Presence of objective vs subjective statements\n\n"
        "Return a float between 0 and 1, where:\n"
        "1.0 = highly trustworthy\n"
        "0.0 = not trustworthy"
    )
    
    # Prompt for relevance scoring
    RELEVANCE_SCORE = (
        "Evaluate the relevance between the query and the candidate text:\n\n"
        "Query: {query}\n"
        "Candidate: {candidate_text}\n\n"
        "Consider:\n"
        "1. Semantic similarity\n"
        "2. Information overlap\n"
        "3. Query intent matching\n"
        "4. Contextual relevance\n\n"
        "Return a float between 0 and 1, where:\n"
        "1.0 = highly relevant\n"
        "0.0 = not relevant"
    )
    
    # Prompt for metadata extraction
    METADATA_EXTRACT = (
        "Extract key metadata from the following text:\n\n"
        "Text: {text}\n\n"
        "Extract:\n"
        "1. Topics mentioned\n"
        "2. Named entities\n"
        "3. Temporal indicators\n"
        "4. Geographic references\n"
        "5. Key concepts\n\n"
        "Return in JSON format with these fields."
    ) 