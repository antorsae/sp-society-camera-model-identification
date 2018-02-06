import argparse
import requests
import urllib.parse
import json
import wget
import os
import glob
from collections import defaultdict

RESOLUTIONS = {                # samples*2  model      samples   EXIF orientation
#
    0: [[2688,1520, 508/550],  # 508        htc_m7         275   Orientation: TopLeft
        [1520,2688,  42/550]], #  42
    1: [[2448,3264,       1]], # 550        iphone_6       273   Orientation: RightTop
                               #                             2   Orientation: TopLeft    
    2: [[2432,4320, 472/550],  # 472        moto_maxx      275   Orientation: Undefined
        [4320,2432,  78/550]], #  78 
    3: [[3120,4160, 468/550],  # 468        moto_x         275   Orientation: Undefined
        [4160,3120,  82/550]],  #  82
        #[2432,4320,      0]], 
    4: [[2322,4128,       1],
        [3096,4128,       0]], # 550        samsung_s4     275   Orientation: RightTop
    5: [[2448,3264,       1]], # 550        iphone 4s      275   Orientation: RightTop
    6: [[4032,3024, 544/550],  # 544        nexus_5x       275   Orientation: Undefined
        [3024,4032,   6/550]], #   6 
    7: [[780, 1040,   2/550],  #   2        nexus_6          1   Orientation: LeftBottom
        [4130,3088,   2/550],  #   2                       274   Orientation: TopLeft
        [4160,3088,  30/550],  #  30
        [4160,3120, 256/550],  # 256
        [3088,4160,  18/550],  #  18
        [3120,4160, 242/550]], # 242
    8: [[2322,4128,       1],
        [3096,4128,       0],
        [2448,3264,       0],
        [1836,3264,       0],
        [1552,2048,       0]], # 550        samsung_note3  196   Orientation: RightTop
                               #                            79   Orientation: TopLeft
    9: [[4000,6000,       1]], # 550        sony_nex7       35   Orientation: LeftBottom
                               #                             3   Orientation: RightTop
                               #                           237   Orientation: TopLeft
}

EXTRA_CLASSES = [
    'htc_m7',
    'iphone_6',
    'moto_maxx',
    'moto_x',
    'samsung_s4',
    'iphone_4s',
    'nexus_5x',
    'nexus_6',
    'samsung_note3',
    'sony_nex7'
]

GROUPS = [
	['1952940@N23'], # htc7
	[],                # iphone_6
	[],                # moto_maxx
	['2437410@N20', '3010976@N23'],                # moto_x
	[],                # samsung_s4
	[],                # iphone_4s
	['2262877@N24', '2984635@N22', '21279683@N00', '2373280@N20',  # Nexus 5x has same camera than Nexus 6P
	 '2621487@N24', '2911955@N21', '2426209@N22' , '2771562@N23',
	 '2528280@N22', '2944106@N22', '2936356@N22'], # nexus_5x
	['2703039@N20'],                               # nexus_6
	[],                # samsung_note3
	[]                 # sony_nex7
]

CAMERAS = [
	['htc/one'],
	['apple/iphone_6'],
	['motorola/moto_maxx', 'motorola/droid_ultra'],
	['motorola/moto_x'],
	['samsung/galaxy_s4'],
	['apple/iphone_4s'],
	[],
	['motorola/nexus_6'],
	['samsung/galaxy-note-3'],
	['sony/nex-7']
]

KEYWORDS = [
	[],
	[],
	[],
	['xt1096', 'XT1096'],
	[],
	[],
	['nexus 5x', 'nexus 6p', 'nexus 5X', 'nexus 6P', 'nexus-5x', 'nexus-6p', 'nexus-5X', 'nexus-6P', 'nexus_5x', 'nexus_6p', 'nexus_5X', 'nexus_6P'],
	[],
	[],
	[]
]

GROUP_URL   = 'https://api.flickr.com/services/rest/?method=flickr.photos.search&api_key={}&license=1%2C2%2C3%2C4%2C5%2C6%2C7%2C8%2C9%2C10&content_type=1&group_id={}&extras=url_o%2Coriginal_format&per_page=500&page={}&format=json&nojsoncallback=1'
CAMERA_URL  = "https://api.flickr.com/services/rest?sort=relevance&api_key={}&parse_tags=1&content_type=7&extras=can_comment%2Ccount_comments%2Ccount_faves%2Cdescription%2Cisfavorite%2Clicense%2Cmedia%2Cneeds_interstitial%2Cowner_name%2Cpath_alias%2Crealname%2Crotation%2Curl_c%2Curl_l%2Curl_m%2Curl_n%2Curl_q%2Curl_s%2Curl_sq%2Curl_t%2Curl_z&per_page=500&lang=en-US&camera={}&page={}&view_all=1&license=1%2C4%2C5%2C6%2C9%2C10&media=photos&method=flickr.photos.search&csrf=1517782385%3A6vq5zvwk8j9%3A3070bf5bcabc4eaa4e0f833b006a212f&format=json&hermes=1&hermesClient=1&nojsoncallback=1&extras=url_o,machine_tags"
KEYWORD_URL = "https://api.flickr.com/services/rest?sort=relevance&api_key={}&parse_tags=1&content_type=7&extras=can_comment%2Ccount_comments%2Ccount_faves%2Cdescription%2Cisfavorite%2Clicense%2Cmedia%2Cneeds_interstitial%2Cowner_name%2Cpath_alias%2Crealname%2Crotation%2Curl_c%2Curl_l%2Curl_m%2Curl_n%2Curl_q%2Curl_s%2Curl_sq%2Curl_t%2Curl_z&per_page=500&lang=en-US&text={}&page={}&view_all=1&license=1%2C4%2C5%2C6%2C9%2C10&media=photos&method=flickr.photos.search&csrf=1517782385%3A6vq5zvwk8j9%3A3070bf5bcabc4eaa4e0f833b006a212f&format=json&hermes=1&hermesClient=1&nojsoncallback=1&extras=url_o,machine_tags"

parser = argparse.ArgumentParser()
# general
parser.add_argument('-a', '--api-key', type=str, default=None, help='Flickr API key')
args = parser.parse_args()

if args.api_key is None:
	if os.path.exists('flickr_api'):
		args.api_key = open('flickr_api','r').read().replace('\n', '')
else:
	print("You need to provide a valid flickr API key, either with -a or just putting the key at ./flickr_api")
	assert False

API_KEY = args.api_key

args = parser.parse_args()

def get_page(page, group_id, url):
	print(group_id)
	group_id = urllib.parse.quote(group_id)
	r = requests.get(url.format(API_KEY, group_id, page))
	d = json.loads(r.content)
	if 'photos' not in d:
		print(d)
		assert False
	d_photos = d['photos']
	return d_photos

def check_resolution(photo, resolutions):
	h,w = int(photo['height_o']), int(photo['width_o'])
	for resolution in resolutions:
		if (h == resolution[0] and w == resolution[1]) or \
			h == resolution[1] and w == resolution[0]:
			return True, (h,w)

	return False, (h,w)

SEARCH_URL  = CAMERA_URL
SEARCH_WHAT = CAMERAS # GROUPS

MAX_PHOTOS = 4000
MAX_PAGES  = 500

for search_what, search_url in [[KEYWORDS, KEYWORD_URL], [CAMERAS, CAMERA_URL], [GROUPS, GROUP_URL]]:
	for class_idx, groups in enumerate(search_what):
		print("Processing {}".format(EXTRA_CLASSES[class_idx]))
		print(groups)
		for group_id in groups:
			print(" group_id {}".format(group_id))

			d_photos = get_page(1, group_id, search_url)

			valid_photos = len(glob.glob(os.path.join(EXTRA_CLASSES[class_idx], '*.jpg')))
			all_photos   = 0
			pages = d_photos['pages']+1
			bad_resolutions= defaultdict(int)
			for page in range(1, min(MAX_PAGES, pages)):
				print("Loading page: {}/{}".format(page, pages))
				if page != 1:
					d_photos = get_page(page, group_id, SEARCH_URL)

				for photo in d_photos['photo']:
					resolution_ok, (h,w) = check_resolution(photo, RESOLUTIONS[class_idx])
					if resolution_ok:
						filename = os.path.join(EXTRA_CLASSES[class_idx], photo['url_o'].split('/')[-1])
						if not os.path.exists(filename):
							wget.download(photo['url_o'], out=EXTRA_CLASSES[class_idx])
							valid_photos += 1
						else:
							#print("Already downloaded: {}".format(filename))
							pass
					else:
						# bad resolution
						bad_resolutions[(h,w)] += 1 
					all_photos += 1
					if valid_photos >= MAX_PHOTOS:
						break

				if valid_photos >= MAX_PHOTOS:
					print("Reached {}/{} photos for {}".format(valid_photos, MAX_PHOTOS, EXTRA_CLASSES[class_idx]))
					break

				print("Got {} photos for {}".format(valid_photos, EXTRA_CLASSES[class_idx]))

			print("Valid photos: {}/{} for {}".format(valid_photos, all_photos, EXTRA_CLASSES[class_idx]))
			print("Ignored resolutions:")
			for res, times in bad_resolutions.items():
				if times > 10:
					print("{}: {} ".format(res, times))




