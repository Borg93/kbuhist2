import pytest

from diachronic.preprocess.sentence_regex import SentRegex


@pytest.mark.parametrize(
    "input_list, expected_output_list",
    [
        (
            [
                "\n",
                "Psykologiska gåtor\n",
                "En rättegång för sextio år sedan.\n",
                "I.\n",
                "I en vacker salon på ett litet landtställe nära floden, ett par mil ifrån London.\n",
                "— Nej, troligen icke, — svarade Astley förvånad och kännande sina misstankar.\n",
                "— Gott Godt!\n",
                "heJ »» HEJ» The     quick brown    fox, Augusta på 	Isa, o c h Isa p å A u g u s t",
                "--",
                "XI",
                "XI -",
                "XI - s",
                "— Godt!\n",
            ],
            [
                "Psykologiska gåtor",
                "En rättegång för sextio år sedan.",
                "I en vacker salon på ett litet landtställe nära floden, ett par mil ifrån London.",
                "— Nej, troligen icke, — svarade Astley förvånad och kännande sina misstankar.",
                "— Gott Godt!",
                "heJ »» HEJ» The quick brown fox, Augusta på Isa, o c h Isa p å A u g u s t",
                "XI",
                "XI -",
                "XI - s",
                "— Godt!",
            ],
        ),
        (
            [
                "hej jag mår bra ...",
                "....",
                "...",
                "; ...",
                "? ...",
                "!...",
                ".....",
                ". . . . .",
                ". . . - ...",
                ";... hej hur mår du ;",
                "—...",
                "— ...",
                "- ...",
            ],
            ["hej jag mår bra ...", "hej hur mår du ;"],
        ),
        (
            [
                "– – hej – –",
                "— — — hej",
                " — — — ",
                " — —  ",
                "– ",
                "– hej",
                "– – –  end och start? remove end and replace start with – – –",
                "– – –  end och start? remove end and replace start with – –",
                "– – –  end och start? remove end and replace start with – – – –",
                " – – –  end och start? remove end and replace start with – – – ",
            ],
            [
                "— hej",
                "— hej",
                "– hej",
                "— end och start? remove end and replace start with",
                "— end och start? remove end and replace start with",
                "— end och start? remove end and replace start with",
                "— end och start? remove end and replace start with",
            ],
        ),
    ],
)
def test_clean_list_from_roman_and_specialchar_and_whitespace(
    input_list, expected_output_list
):
    given_sub_tuple = (
        (
            r"^[\.]{2,5}|[;]\.{2,5}|(\.\s){2,5}|[\.]{4,5}|[;(?!)]\.{2,5}|[;(?!)] \.{2,5}",
            " ",
        ),
        (r"[-—] \.{2,5}|[-—]\.{2,5}", "— "),
        (r"^(–\s){2,5}|^(\s–){2,5}|^(—\s){2,5}|^(\s—){2,5}", "— "),
        (r"(–\s){2,5}$|(\s–){2,5}$|(—\s){,5}$|(\s—){2,5}$", " "),
        (r"(-\s){2,5}|(—\s){2,5}", " "),
        (r"(–\s){2,5}|(—\s){2,5}", " "),
        (r"(-){2,5}|(—){2,5}", " "),
        (r"(?<=[a-zA-Z])-\s|(?<=[a-zA-Z])—\s", ""),
        (r",,", ","),
        (r"\s+", " "),
        (r"\t", " "),
        (r"[ \t]+$", ""),
        (r"^\s+", ""),
    )
    sent_regex = SentRegex(
        sub_tuple=given_sub_tuple, remove_starting_roman_chapters=True
    )

    assert (
        sent_regex.clean_list_from_roman_and_specialchar_and_whitespace(input_list)
        == expected_output_list
    )
