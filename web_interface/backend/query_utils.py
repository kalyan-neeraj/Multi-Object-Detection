import datetime

def mock_query_processing(query, detected_objects):
    q = query.lower()
    if "age" in q and "2000-01-01" in q:
        birth_date = datetime.date(2000, 1, 1)
        today = datetime.date.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return f"A person born on January 1, 2000 is {age} years old today."
    if "red" in q and "cube" in q:
        return "There is 1 red cube in the image. It is located in the left portion of the image and is made of plastic."
    if "blue" in q and "sphere" in q:
        return "There is 1 blue sphere in the image. It is positioned to the right of the red cube and is made of metal."
    if "how many" in q:
        return "There are {} objects detected in the image.".format(len(detected_objects))
    return "Based on my analysis of the image, I can see various geometric shapes in different colors and sizes. Could you please be more specific about what spatial relationships you'd like me to describe?"
