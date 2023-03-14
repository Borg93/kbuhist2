import pytest
from kubhist.prepreprocessing_khubist import Clean_Khubis


test_sent = ['\n', 
             'Psykologiska gåtor\n',
             'En rättegång för sextio år sedan.\n',
             'I.\n',
             'I en vacker salon på ett litet landtställe nära floden, ett par mil ifrån London, sutto på eftermiddagen, samma dag som Alf och Malkolm skildes åt i Kranach, tvenne fruntimmer.\n', 
              '— Nej, troligen icke, — svarade Astley förvånad och kännande sina misstankar, som nyss varit helt och hållet försvunna, åter vakna.\n',
                          '— Gott Godt!\n',
             "heJ »» HEJ» The     quick brown    fox, Augusta på 	Isa, o c h Isa p å A u g u s t",
             "--",
             "XI",
             "XI -"
             "XI - s"
             '— Godt!\n']


def test_something():
    # GIVEN a mobile is registered
    ... some source code ...

    # WHEN a test mode data connection is initiated
    ... some source code ...

    # THEN the call should connect
    ... some source code ...

######################


^[\.]{2,5}|[;]\.{2,5}|(\.\s){2,5}|[\.]{4,5}|[;(?!)]\.{2,5}|[;(?!)] \.{2,5}

____

hej jag mår bra ...
....
...
; ...
? ...
!...
..... 
. . . . .
. . . - ...
;... hej hur mår du ;
____

hej jag mår bra ...

- ...
################


[-—] \.{2,5}|[-—]\.{2,5}
#########
—...
— ...
- ...
#######
—

#########
^(–\s){2,5}|^(\s–){2,5}|^(—\s){2,5}|^(\s—){2,5}
#########
– – hej – –
— — — hej
 — — — 
 — —
– – –  end och start? remove end and replace start with – – –
 – – –  end och start? remove end and replace start with – – – 

#########

—hej – –
—hej
— 
—
— end och start? remove end and replace start with – – –
—  end och start? remove end and replace start with – – – 

####
(–\s){2,5}$|(\s–){2,5}$|(—\s){,5}$|(\s—){2,5}$
###########

– – hej – –
— — — hej
 — — — 
 — —  
–  
– hej
– – –  end och start? remove end and replace start with – – –
– – –  end och start? remove end and replace start with – –
– – –  end och start? remove end and replace start with – – – –
 – – –  end och start? remove end and replace start with – – – 

########

– – hej
— — — hej
 
 — —  
–  
– hej
– – –  end och start? remove end and replace start with
– – –  end och start? remove end and replace start with
– – –  end och start? remove end and replace start with
 – – –  end och start? remove end and replace start with 