from evaluate import load

predictions1 = ["in his memory", "in memory of him", "in remembrance of him",
               "in her memory", "in memory of her", "in remembrance of her"]

predictions2 = [
    "from two days ago",
    "from two weeks ago",
    "from two months ago",
    "from two years ago",
    "from seventeen falls ago",
    "from seventeen autumns ago",
    "from twenty weeks ago",
    "from many years ago",
    "from three moons ago",
    "from many moons ago",
    "from century ago",
    "from last week",
    "from last month",
    "from last year",
    "from last August",
    "from last century",
    "from last decade",
    "from the last week",
    "from the last month",
    "from the last year",
    "from the last August",
    "from the last century",
    "from the last decade"
]

predictions3 = [
    "[Original] story",
    "(Original) story",
    "Original story",
    "[Abridged] story",
    "(Abridged) story",
    "Abridged story",
     "abridged story",
    "[PERSON] in",
    "[PERSON11] in",
    "[person] in",
    "PERSON in",
    "person in",
    "Person in",
    "(CAT) in",
    "[CAT] in",
    "cat in",
    "[HIPPO] in",
    "hippo in"
]

predictions4 = [
    "dog's hair",
    "hair of a dog",
    "dog's tail",
    "tail of a dog",

    "dog's hair",
    "hair of dogs",
    "hair of dog",
    "food for dogs",
    "food for dog",
    "dog's food",
    "dog food",
    "get some dog food",
    "get some",
    "dog's loyalty",
    "loyalty of dog"
]

model_id = 'gpt2'
# model_id = 'EleutherAI/gpt-j-6B''EleutherAI/gpt-j-6B'
# model_id = 'google-t5/t5-small'
# model_id = 'google-bert/bert-base-cased'

perplexity = load("perplexity", module_type="metric")
results = perplexity.compute(predictions=predictions, model_id=model_id, add_start_token=False)

for pred, res in zip(predictions1, results['perplexities']):
    print(f"{pred:<30} {round(res):>5}")
