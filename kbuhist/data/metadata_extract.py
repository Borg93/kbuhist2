import logging

import requests
from bs4 import BeautifulSoup


def get_all_info_of_table(corpus="letters"):

    corpuses_id_dict = {
        "letters": 0,
        "court": 1,
        "school": 2,
        "informal": 3,
        "law": 4,
        "governmental": 5,
        "pamphlets": 6,
        "religion": 7,
        "secular": 8,
        "user-generated": 9,
        "lyrics": 10,
        "newspapers": 11,
        "periodicals": 12,
        "academic-scientific": 13,
    }

    URL = "https://cl.lingfil.uu.se/svediakorp/index.html"
    r = requests.get(URL)
    soup = BeautifulSoup(r.content, "html.parser")

    tables = soup.findAll("table")

    header = soup.find_all("h3")

    heading = header[corpuses_id_dict[corpus]].get_text()
    table = tables[corpuses_id_dict[corpus]]

    corpus_dataset_date_list = []

    for row in table.find_all("tr")[1:]:
        columns = row.find_all("td")
        if len(columns) > 0:
            dataset = columns[0].get_text()
            date = columns[1].get_text()
            file_name = columns[3].find('a').get('href')

            corpus_dataset_date_list.append((heading, corpus, dataset, date, file_name))
    
    return sorted(corpus_dataset_date_list, key=lambda x: x[-1])


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # assemble_parquet()
    # TODO Try to find and add metadata to groups

    corpus_dataset_date_list = get_all_info_of_table("court")

    print(corpus_dataset_date_list)
