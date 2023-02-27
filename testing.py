# test_list = ["De s o m f o r s k a t efter h e n n e s l e f n a d s ö d e n h a e n d t u p p t ä c k t, att h o n v a r i t d o t t e r af l a n d s k a m r e r a r e n i Örebro län, A n d e r s W a r g, och Katharina L e w i n — hvilken senare blef omgift m e d en ryttmästare Eosenstråle.",
# " — v a r f ö d d d e n 2 3 m a r s 1 7 0 3 o c h h e t t e Kri- stina — h u r n a m n e t K a j s a u p p k o m m i t, är s v å r t att s ä g a, så m y c k e t m e r s o m h o n själf i 4:" ,
# "d e u p p l a g a n af sin b e r ö m d a k o k - b o k k a l l a r s i g v i d, det r ä t t a n a m n e t: K r i - stina. ",
# "Hennes » Hjälpreda i hushållningen för u n g a f r u e n t i m b e r », tillägnad » n å d i g a o c h gunstiga fruer », trycktes första g å n g e n 1 7 5 5 o c h h a r s e d a n, till 1 8 4 9, u t g å t t i 1 5 s v e n - ska upplagor o c h minst 4 t y s k a:" ,
# "— ingen s v e n s k k v i n n a har sett sin litterära verk- s a m h e t så u p p m ä r k s a m m a d o c h m å n g f a l d i - d i g a d, m e d u n d a n t a g af — d e n h e l i g a Bir- gitta!",
# "— ingen s v e n s k k v i n n a har sett sin litterära verk-å Bir- gitta!",
# "— ingen s v e n s k k v har sett sin litterära verk-å Bir- gitta!",
# "— ingen s v e n  verk-å Bir- gitta!",
# "De s o m f o r s k a t efter h e n n e s l e f n a d s ö d e n h a e n d t u p p t ä c k t, att h o n v a r i t d o t t e r af l a n d s k a m r e r a r e n i Örebro län, A n d e r s W a r g, och Katharina L e w i n — hvilken senare blef omgift m e d en ryttmästare Eosenstråle — v a r f ö d d d e n 2 3 m a r s 1 7 0 3 o c h h e t t e Kri- stina — h u r n a m n e t K a j s a u p p k o m m i t, är s v å r t att s ä g a, så m y c k e t m e r s o m h o n själf i 4: d e u p p l a g a n af sin b e r ö m d a k o k - b o k k a l l a r s i g v i d, det r ä t t a n a m n e t: K r i - stina. Hennes » Hjälpreda i hushållningen för u n g a f r u e n t i m b e r », tillägnad » n å d i g a o c h gunstiga fruer », trycktes första g å n g e n 1 7 5 5 o c h h a r s e d a n, till 1 8 4 9, u t g å t t i 1 5 s v e n - ska upplagor o c h minst 4 t y s k a: — ingen s v e n s k k v i n n a har sett sin litterära verk- s a m h e t så u p p m ä r k s a m m a d o c h m å n g f a l d i - d i g a d, m e d u n d a n t a g af — d e n h e l i g a Bir- gitta!",
# " Hej jag mår jättebra idag hur mår du i denna strålande dag som i i idag",
# "D e t f o r d r a s s å l e d e s e n s t o r f o n d af f y s i s k kraft att t a g a af, e t t b r e d t u n d e r l a g af k r o p p s l i g s t y r k a o c h u t b i l d n i n g f ö r att u p p - b ä r a o c h g e u t t r y c k åt d e n a n d l i g a t a l a n g e n. D e t är i v å r t i d p å s ä t t o c h v i s m o d e r n t att t a l a o m a n d l i g m a k t ö f v e r m ä n n i s k o r i m o t s a t s till o c h m e d f r å n s e e n d e af f y s i s k a h j ä l p m e d e l till d e s s f ö r v e r k l i g a n d e, o m fjärr- v e r k a n o. s. v. D ä r o m veta v i d o c k p l a t t intet. O c h analysera vi, hur vi verka på a n d r a eller s j ä l f v a p å v e r k a s, så är d e t ej s v å r t att finna d e m a t e r i e l l a h ä f s t ä n g e r n a. F y s i s k k r a f t s å l u n d a f ö r att g ö r a d e n a n d - liga fruktbärande!",
# "I ",
# "I i"]


# def counting_lenght_of_letters_and_if_to_many_remove(sent_list):
#     new_sent_list = []
#     for sent in sent_list:
#         splitted_sent = sent.split()
#         counter_word_length = {"long": 0, "short":0}
#         for word in splitted_sent:
#             if len(word)  > 1:
#                 counter_word_length["long"] +=1
#             else:
#                 counter_word_length["short"] +=1
#         counter_ratio = (counter_word_length["long"]+0.5) / (counter_word_length["short"]+0.001) 
#         counter_ratio_len = 1- (counter_word_length["short"] / len(splitted_sent))
#         counter_avg = (counter_ratio+counter_ratio_len)/2

#         if counter_avg > 0.65 or 0> len(splitted_sent) < 4:
#             new_sent_list.append(sent)
#     return new_sent_list


# new_test_list= counting_lenght_of_letters_and_if_to_many_remove(test_list)
# print(new_test_list)
from datasets import load_dataset, Dataset, DatasetDict


dataset = load_dataset("Gabriel/mini_khubist2_v2", split="train").to_pandas()
dataset.to_csv("test.csv")
