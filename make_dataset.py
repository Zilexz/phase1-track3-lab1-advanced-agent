"""Sinh bộ dữ liệu benchmark multi-hop đa dạng (>=120 ví dụ).

Mỗi ví dụ là SELF-CONTAINED: mọi fact cần thiết đều nằm trong `context`, nên
Actor (chỉ dùng context) luôn có thể trả lời đúng, và đáp án không phụ thuộc
kiến thức ngoài. Bộ này gồm 8 câu hotpot gốc + nhiều họ câu hỏi 2-hop khác nhau.

Chạy:  python make_dataset.py            # -> data/benchmark.json
"""
from __future__ import annotations

import json
from pathlib import Path

# (capital, country, water-body, continent, currency)
COUNTRIES = [
    ("Paris", "France", "Atlantic Ocean", "Europe", "the euro"),
    ("Madrid", "Spain", "Mediterranean Sea", "Europe", "the euro"),
    ("Rome", "Italy", "Mediterranean Sea", "Europe", "the euro"),
    ("Lisbon", "Portugal", "Atlantic Ocean", "Europe", "the euro"),
    ("Berlin", "Germany", "Baltic Sea", "Europe", "the euro"),
    ("Oslo", "Norway", "North Sea", "Europe", "the Norwegian krone"),
    ("Stockholm", "Sweden", "Baltic Sea", "Europe", "the Swedish krona"),
    ("Amsterdam", "the Netherlands", "North Sea", "Europe", "the euro"),
    ("Athens", "Greece", "Aegean Sea", "Europe", "the euro"),
    ("Warsaw", "Poland", "Baltic Sea", "Europe", "the Polish zloty"),
    ("Reykjavik", "Iceland", "Atlantic Ocean", "Europe", "the Icelandic krona"),
    ("Helsinki", "Finland", "Baltic Sea", "Europe", "the euro"),
    ("Cairo", "Egypt", "Red Sea", "Africa", "the Egyptian pound"),
    ("Nairobi", "Kenya", "Indian Ocean", "Africa", "the Kenyan shilling"),
    ("Tokyo", "Japan", "Pacific Ocean", "Asia", "the Japanese yen"),
    ("Hanoi", "Vietnam", "South China Sea", "Asia", "the Vietnamese dong"),
    ("New Delhi", "India", "Indian Ocean", "Asia", "the Indian rupee"),
    ("Bangkok", "Thailand", "Gulf of Thailand", "Asia", "the Thai baht"),
    ("Amman", "Jordan", "Red Sea", "Asia", "the Jordanian dinar"),
    ("Lima", "Peru", "Pacific Ocean", "South America", "the Peruvian sol"),
    ("Santiago", "Chile", "Pacific Ocean", "South America", "the Chilean peso"),
    ("Brasilia", "Brazil", "Atlantic Ocean", "South America", "the Brazilian real"),
    ("Buenos Aires", "Argentina", "Atlantic Ocean", "South America", "the Argentine peso"),
    ("Mexico City", "Mexico", "Pacific Ocean", "North America", "the Mexican peso"),
    ("Ottawa", "Canada", "Atlantic Ocean", "North America", "the Canadian dollar"),
]

# country -> (official language, language family)
LANGUAGES = {
    "France": ("French", "Romance"),
    "Spain": ("Spanish", "Romance"),
    "Italy": ("Italian", "Romance"),
    "Portugal": ("Portuguese", "Romance"),
    "Brazil": ("Portuguese", "Romance"),
    "Germany": ("German", "Germanic"),
    "Norway": ("Norwegian", "Germanic"),
    "Sweden": ("Swedish", "Germanic"),
    "the Netherlands": ("Dutch", "Germanic"),
    "Iceland": ("Icelandic", "Germanic"),
    "Greece": ("Greek", "Hellenic"),
    "Poland": ("Polish", "Slavic"),
    "Egypt": ("Arabic", "Semitic"),
    "Jordan": ("Arabic", "Semitic"),
    "Japan": ("Japanese", "Japonic"),
    "Vietnam": ("Vietnamese", "Austroasiatic"),
    "Finland": ("Finnish", "Uralic"),
    "India": ("Hindi", "Indo-Aryan"),
}

# (author, book, university, birth_city, river-through-city)
AUTHORS = [
    ("J. R. R. Tolkien", "The Hobbit", "Oxford University", "Bloemfontein", "the Modder River"),
    ("C. S. Lewis", "The Chronicles of Narnia", "Cambridge University", "Belfast", "the River Lagan"),
    ("Lewis Carroll", "Alice in Wonderland", "Oxford University", "Daresbury", "the River Mersey"),
    ("Umberto Eco", "The Name of the Rose", "the University of Bologna", "Alessandria", "the Tanaro River"),
    ("J. K. Rowling", "Harry Potter", "the University of Exeter", "Yate", "the River Frome"),
]

# (composer, work, instrument)
COMPOSERS = [
    ("Antonio Vivaldi", "The Four Seasons", "violin"),
    ("Frederic Chopin", "the Nocturnes", "piano"),
    ("Niccolo Paganini", "the 24 Caprices", "violin"),
    ("Wolfgang Amadeus Mozart", "the Clarinet Concerto", "piano"),
    ("Pablo Casals", "the Bach Cello Suites recordings", "cello"),
    ("Louis Armstrong", "What a Wonderful World", "trumpet"),
]

# (painter, painting, museum, museum_city)
PAINTINGS = [
    ("Leonardo da Vinci", "the Mona Lisa", "the Louvre", "Paris"),
    ("Vincent van Gogh", "The Starry Night", "the Museum of Modern Art", "New York City"),
    ("Johannes Vermeer", "Girl with a Pearl Earring", "the Mauritshuis", "The Hague"),
    ("Diego Velazquez", "Las Meninas", "the Museo del Prado", "Madrid"),
    ("Sandro Botticelli", "The Birth of Venus", "the Uffizi Gallery", "Florence"),
]

# (scientist, achievement, second_field)
SCIENTISTS = [
    ("Isaac Newton", "the three laws of motion", "mathematics"),
    ("Albert Einstein", "the theory of relativity", "philosophy of science"),
    ("Marie Curie", "research on radioactivity", "chemistry"),
    ("Blaise Pascal", "Pascal's principle of fluid pressure", "mathematics"),
    ("Alan Turing", "the Turing machine concept", "cryptanalysis"),
]

HOTPOT = json.loads(Path("data/hotpot_mini.json").read_text(encoding="utf-8"))

# Câu KHÓ thủ công: 3-hop có "đường tắt" cám dỗ, distractor cùng loại, phủ định,
# so sánh, thứ tự (superlative). Mục tiêu: khiến cả model mạnh thỉnh thoảng sai
# theo NHIỀU kiểu khác nhau -> báo cáo thật có >=3 failure mode. Vẫn self-contained.
HARD = [
    {"q": "What river flows through the capital of the country where the composer Edvard Grieg was born?", "a": "Akerselva", "ctx": [
        ("Edvard Grieg", "Edvard Grieg, the composer, was born in the city of Bergen."),
        ("Bergen", "Bergen lies beside the Byfjorden inlet."),
        ("Norway", "The capital of Norway is Oslo."),
        ("Oslo", "The Akerselva river flows through Oslo.")]},
    {"q": "Among the countries Austria borders, which one does NOT use the euro?", "a": "Switzerland", "ctx": [
        ("Austria", "Austria borders Germany, Italy, and Switzerland."),
        ("Germany", "Germany uses the euro."),
        ("Italy", "Italy uses the euro."),
        ("Switzerland", "Switzerland uses the Swiss franc, not the euro.")]},
    {"q": "Which river is the second longest in the list of South American rivers below?", "a": "Parana", "ctx": [
        ("Amazon", "The Amazon is about 6400 km long."),
        ("Parana", "The Parana is about 4880 km long."),
        ("Madeira", "The Madeira is about 3250 km long.")]},
    {"q": "In which mountain range is the highest peak of the country whose capital is Quito?", "a": "the Andes", "ctx": [
        ("Quito", "Quito is the capital of Ecuador and sits near the Pichincha volcano."),
        ("Ecuador", "The highest peak in Ecuador is Chimborazo."),
        ("Chimborazo", "Chimborazo is part of the Andes.")]},
    {"q": "Between the authors of 'War and Peace' and 'Crime and Punishment', who was born first?", "a": "Fyodor Dostoevsky", "ctx": [
        ("War and Peace", "War and Peace was written by Leo Tolstoy, who was born in 1828."),
        ("Crime and Punishment", "Crime and Punishment was written by Fyodor Dostoevsky, who was born in 1821.")]},
    {"q": "Which of these three scientists did NOT win a Nobel Prize: Marie Curie, Albert Einstein, or Dmitri Mendeleev?", "a": "Dmitri Mendeleev", "ctx": [
        ("Marie Curie", "Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911."),
        ("Albert Einstein", "Albert Einstein won the Nobel Prize in Physics in 1921."),
        ("Dmitri Mendeleev", "Dmitri Mendeleev was never awarded a Nobel Prize.")]},
    {"q": "How many years passed between the founding of Apple and the release of the first iPhone?", "a": "31 years", "ctx": [
        ("Apple", "Apple was founded in 1976."),
        ("iPhone", "Apple released the first iPhone in 2007.")]},
    {"q": "What is the capital of the country that hosted the 2016 Summer Olympics?", "a": "Brasilia", "ctx": [
        ("2016 Summer Olympics", "The 2016 Summer Olympics were held in Rio de Janeiro."),
        ("Rio de Janeiro", "Rio de Janeiro is a major city in Brazil, but it is not the capital."),
        ("Brazil", "The capital of Brazil is Brasilia.")]},
    {"q": "What is the capital of the country whose largest city is Sydney?", "a": "Canberra", "ctx": [
        ("Sydney", "Sydney is the largest city in Australia, but it is not its capital."),
        ("Australia", "The capital of Australia is Canberra.")]},
    {"q": "What is the capital of the country whose largest city is Istanbul?", "a": "Ankara", "ctx": [
        ("Istanbul", "Istanbul is the largest city of Turkey, though it is not the capital."),
        ("Turkey", "The capital of Turkey is Ankara.")]},
    {"q": "Which sea borders the western coast of the country whose largest city is Mumbai?", "a": "the Arabian Sea", "ctx": [
        ("Mumbai", "Mumbai is the largest city of India."),
        ("India", "India's western coast lies along the Arabian Sea, while its eastern coast lies along the Bay of Bengal.")]},
    {"q": "Which of these countries does NOT border France: Spain, Germany, or Portugal?", "a": "Portugal", "ctx": [
        ("France", "France borders both Spain and Germany."),
        ("Portugal", "Portugal borders only Spain.")]},
    {"q": "What is the longest river on the continent where Egypt is located?", "a": "the Nile", "ctx": [
        ("Egypt", "Egypt is located on the continent of Africa."),
        ("Africa", "The longest river in Africa is the Nile; the Congo is the second longest.")]},
    {"q": "In the list below, which mountain is the third highest?", "a": "Kangchenjunga", "ctx": [
        ("Mount Everest", "Mount Everest is 8849 m tall."),
        ("K2", "K2 is 8611 m tall."),
        ("Kangchenjunga", "Kangchenjunga is 8586 m tall."),
        ("Lhotse", "Lhotse is 8516 m tall.")]},
    {"q": "Which was completed first: the Eiffel Tower or the Statue of Liberty?", "a": "the Statue of Liberty", "ctx": [
        ("Statue of Liberty", "The Statue of Liberty was completed in 1886."),
        ("Eiffel Tower", "The Eiffel Tower was completed in 1889.")]},
    {"q": "Who directly succeeded the first President of the United States?", "a": "John Adams", "ctx": [
        ("George Washington", "George Washington was the first President of the United States."),
        ("John Adams", "John Adams directly succeeded George Washington as president."),
        ("Thomas Jefferson", "Thomas Jefferson became the third president, after John Adams.")]},
    {"q": "Which ocean borders the eastern coast of the country whose capital is Wellington?", "a": "the Pacific Ocean", "ctx": [
        ("Wellington", "Wellington is the capital of New Zealand."),
        ("New Zealand", "New Zealand's eastern coast faces the Pacific Ocean, while the Tasman Sea lies to its west.")]},
    {"q": "Which of these three languages is NOT a Romance language: Romanian, Romansh, or Romani?", "a": "Romani", "ctx": [
        ("Romanian", "Romanian is a Romance language."),
        ("Romansh", "Romansh is a Romance language spoken in Switzerland."),
        ("Romani", "Romani is an Indo-Aryan language, not a Romance language.")]},
    {"q": "What instrument was the composer of the opera 'The Magic Flute' a virtuoso on?", "a": "piano", "ctx": [
        ("The Magic Flute", "The Magic Flute is an opera composed by Wolfgang Amadeus Mozart."),
        ("Wolfgang Amadeus Mozart", "Mozart was a virtuoso on the piano.")]},
    {"q": "What is the capital of the country whose currency is the yen, not the city with the largest population there?", "a": "Tokyo", "ctx": [
        ("Yen", "The yen is the official currency of Japan."),
        ("Japan", "The capital of Japan is Tokyo, which is also its largest city.")]},
]


def chunk(title: str, text: str) -> dict:
    return {"title": title, "text": text}


def build() -> list[dict]:
    out: list[dict] = list(HOTPOT)  # giữ 8 câu gốc để mock vẫn lộ >=3 failure mode
    n = 0

    def add(difficulty: str, question: str, gold: str, ctx: list[dict]) -> None:
        nonlocal n
        n += 1
        out.append({"qid": f"gen{n:03d}", "difficulty": difficulty, "question": question, "gold_answer": gold, "context": ctx})

    for capital, country, water, continent, currency in COUNTRIES:
        cap_ctx = chunk(capital, f"{capital} is the capital city of {country}.")
        add(
            "medium",
            f"Which body of water borders the country whose capital is {capital}?",
            water,
            [cap_ctx, chunk(country, f"{country} borders the {water}.")],
        )
        add(
            "medium",
            f"On which continent is the country whose capital is {capital} located?",
            continent,
            [cap_ctx, chunk(country, f"{country} is located in {continent}.")],
        )
        add(
            "easy",
            f"What is the official currency of the country whose capital is {capital}?",
            currency,
            [cap_ctx, chunk(country, f"The official currency of {country} is {currency}.")],
        )

    for country, (language, family) in LANGUAGES.items():
        add(
            "medium",
            f"Which language family does the official language of {country} belong to?",
            family,
            [
                chunk(country, f"The official language of {country} is {language}."),
                chunk(f"{language} language", f"{language} is a {family} language."),
            ],
        )

    for author, book, university, city, river in AUTHORS:
        add(
            "medium",
            f"Which university did the author of {book} teach at?",
            university,
            [chunk(author, f"{author} wrote {book} and taught at {university}."), chunk(book, f"{book} was written by {author}.")],
        )
        add(
            "hard",
            f"What river flows through the city where the author of {book} was born?",
            river,
            [
                chunk(author, f"{author}, the author of {book}, was born in {city}."),
                chunk(city, f"{city} is crossed by {river}."),
            ],
        )

    for composer, work, instrument in COMPOSERS:
        add(
            "medium",
            f"What instrument did the composer of {work} mainly play?",
            instrument,
            [chunk(work, f"{work} was composed by {composer}."), chunk(composer, f"{composer} mainly played the {instrument}.")],
        )

    for painter, painting, museum, city in PAINTINGS:
        add(
            "hard",
            f"In which city is the museum that houses {painting} located?",
            city,
            [
                chunk(painting, f"{painting} was painted by {painter} and is housed in {museum}."),
                chunk(museum, f"{museum} is located in {city}."),
            ],
        )

    for scientist, achievement, field in SCIENTISTS:
        add(
            "medium",
            f"In addition to physics, what field did the scientist behind {achievement} work in?",
            field,
            [
                chunk(achievement, f"{achievement} is credited to {scientist}."),
                chunk(scientist, f"Besides physics, {scientist} also worked in {field}."),
            ],
        )

    for item in HARD:
        add("hard", item["q"], item["a"], [chunk(t, txt) for t, txt in item["ctx"]])

    return out


def main() -> None:
    data = build()
    path = Path("data/benchmark.json")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(data)} examples -> {path}")


if __name__ == "__main__":
    main()
