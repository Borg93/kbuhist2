import pytest

from diachronic.preprocess.paragraph_chunker import ParagraphChunker


@pytest.fixture
def chunker():
    return ParagraphChunker(chunk_size=128)


@pytest.fixture
def data():
    return [
        "Hon hade tidigt blifvit gift, aldrig älskat sin man, men likväl aldrig känt ett ögonblicks frestelse att älska någon annan, hon var mycket omtyckt, men hade aldrig väckt kärlek i något manligt hjerta och egde ingen qvinlig vän i detta ords egentliga betydelse, ehuru alla hennes bekanta voro förtjusta af hennes sällskap, hon hade aldrig blifvit afundad, aldrig förtalad och var nu, till lynne och känslor vid fyrtiotvå år, precis densamma, som hon varit vid sexton, ett urverk, som aldrig råkat i olag, och hvars förträffliga mekanik tycktes kunna bli ett 'perpetuum mobile'. ",
        "Den andra var hennes dotter, mrs Arabella Fulton, samma vackra fruntimmer, som väckt de unga herrarnes beundran och varit föremål för deras anmärkningar, då hon åkte förbi det lilla utvärdshuset Putney för ett par månader sedan; äfven hon var enka, ehuru ännu icke tjugotvå år. ",
        "Hon tycktes vid första ögonkastet likna sin mor både till utseende och sätt; det var samma vackra blonda hår, samma stora ögon, samma fina hy och vackra växt och äfven hennes något långsamma, lugna och passionsfria sätt, men då man betraktade henne närmare, fann man, att allt detta var, så att säga, ett förräderi. ",
        "Arabella var den fullkomligaste motsats till modern, eller rättare sagdt, det intryck de båda qvinnorna gjorde, var helt och hållet motsatt, ty inom Arabellas halfslutna ögonlock fanns en brinnande elektrisk eld, och öfver hela denna till utseendet så flegmatiska varelse låg utbredt det sällsamma fludium, den hemliga magi, som genast, utan besinning, utan medvetet skäl, med ens och ovilkorligt bringar det andra könet i eld och lågor. ",
        "Hvari består denna hemliga kraft, hvad är orsaken till denna förtrollning, som meddelar sig lika hastigt som blixten, och hvars eld är lika oemotståndlig, lika koncentrerad som den? ",
        "Hvem har någonsin kunnat definiera detta tvetydiga företräde, som vissa qvinnor ega, denna hemliga faddergåfva, som Venus stundom ger sina älsklingar? ",
        "Var Arabella väl så underbart skön? ",
    ]


def test_chunker_split(chunker, data):
    chunks = chunker.chunker_split(data)
    assert len(chunks) == 5


@pytest.mark.parametrize("input_list", [[1, "hello", "world"], "not a list"])
@pytest.mark.xfail(raises=ValueError)
def test_raise_chunker_split(chunker, input_list):
    chunker.chunker_split(input_list)
