import pytest

from kbuhist.preprocess.word_cleaner import WordCleaner


def test_counting_sequence_length_of_numbers():
    cleaner = WordCleaner()

    test_numb_length = [
        "267864,117 580 254 760 14 4 — 92 — 8 32 122 32 ' 74 24 500 126 •_ 400 — 252 800 lä ä — 95 — 10 — 126 32 85",
        "155 — 420 — 270",
        "01 Psykologiska gåtor",
        "Smör 790 588 486 1,087 2,462 5,415",
        "Den 23 April e. m.",
        "hej",
        "Utgående balans: 2,477 375,740",
        "Summa 123,701,90519] 10,401^202 30| 6,462,552 64] 140,565,660] 13 Kontant behållning Inventariers värden af spanmål",
    ]

    expected_output = ["01 Psykologiska gåtor", "Den 23 April e. m.", "hej"]

    assert (
        cleaner.counting_sequence_length_of_numbers(test_numb_length) == expected_output
    )


def test_counting_length_of_letters_and_if_to_many_remove():
    cleaner = Word_Cleaner()

    test_sent_length = [
        "Psykologiska g å t o r hej hej ",
        "En q w e r t q e t u q t h d s  rättegång för sextio år sedan.\n",
        "I en vacker salon på ett litet l a n d t s t ä l l e nära floden, ett par mil ifrån London",
        "— Nej, troligen icke, — svarade Astley förvånad och kännande sina misstankar.",
        "— Gott Godt!",
        "heJ »» HEJ» The     quick brown    fox, Augusta på 	Isa, o c h Isa p å A u g u s t",
        "I en vac k e r s a l o n på e t t l i t e t l a n d t s t älle nära floden, ett par mil ifrån London.",
        "— God t !",
    ]

    expected_output = [
        "I en vacker salon på ett litet l a n d t s t ä l l e nära floden, ett par mil ifrån London",
        "— Nej, troligen icke, — svarade Astley förvånad och kännande sina misstankar.",
        "— Gott Godt!",
        "heJ »» HEJ» The     quick brown    fox, Augusta på \tIsa, o c h Isa p å A u g u s t",
        "— God t !",
    ]

    assert (
        cleaner.counting_length_of_letters_and_if_to_many_remove(test_sent_length)
        == expected_output
    )
