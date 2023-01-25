# Catalog of useful prompt ideas

## Output Format Prompts

```
Given the following transcript of a single person talking, create a summary doc that contains the following -
Summary: 1-2 sentences describing what the person was talking about
Action Items: A list of action items the person said

Transcript: {transcript}
```


## Search Prompts
```
Generate a comprehensive and informative answer (but no more than 80 words) for a given
question solely based on the provided search results. You must only use information from the provided search results. Use an unbiased and journalistic tone. Use the current date, {date}. Combine search results together into a coherent answer, do not repeat text. Only cite the most relevant results that answer the question accurately. If different results refer to different entities with the same name, write separate answers for each entity. 
```