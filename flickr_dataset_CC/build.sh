parallel --eta -N 2 --verbose ./filter_jpgs.sh \'{1}\' \'{2}\' ::: htc_m7 'HTC One' iphone_4s 'iPhone 4S' iphone_6 'iPhone 6' moto_maxx XT1080 moto_x XT1096 nexus_5x 'Nexus (5X|6P)' nexus_6 'Nexus 6' samsung_note3 SM-N9005 samsung_s4 GT-I9505 sony_nex7 NEX-7

cat htc_m7_jpgs iphone_4s_jpgs iphone_6_jpgs moto_maxx_jpgs moto_x_jpgs nexus_5x_jpgs nexus_6_jpgs samsung_note3_jpgs samsung_s4_jpgs sony_nex7_jpgs > good_jpgs
cat *_bad > bad_jpgs
