NYU40_Label_Names = [
    'wall', # 1
    'floor', # 2
    'cabinet', # 3
    'bed', # 4
    'chair', # 5
    'sofa', # 6
    'table', # 7
    'door', # 8 
    'window', # 9
    'bookshelf', # 10
    'picture', # 11
    'counter', # 12
    'blinds', # 13
    'desk', # 14
    'shelves', # 15
    'curtain', # 16
    'dresser', # 17
    'pillow', # 18
    'mirror', # 19
    'floor mat', # 20
    'clothes', # 21
    'ceiling', # 22
    'books', # 23
    'refridgerator', # 24
    'television', # 25
    'paper', # 26
    'towel', # 27
    'shower curtain', # 28
    'box', # 29
    'whiteboard', # 30
    'person', # 31
    'night stand', # 32
    'toilet', # 33
    'sink', # 34
    'lamp', # 35
    'bathtub', # 36
    'bag', # 37
    'otherstructure', # 38
    'otherfurniture', # 39
    'otherprop', # 40
]

def get_NYU40_color_palette() -> list:
    """Get the NYU40 color palette."""
    return [
        (0, 0, 0),
        (174, 199, 232),		# wall
        (152, 223, 138),		# floor
        (31, 119, 180), 		# cabinet
        (255, 187, 120),		# bed
        (188, 189, 34), 		# chair
        (140, 86, 75),  		# sofa
        (255, 152, 150),		# table
        (214, 39, 40),  		# door
        (197, 176, 213),		# window
        (148, 103, 189),		# bookshelf
        (196, 156, 148),		# picture
        (23, 190, 207), 		# counter
        (178, 76, 76),  
        (247, 182, 210),		# desk
        (66, 188, 102), 
        (219, 219, 141),		# curtain
        (140, 57, 197), 
        (202, 185, 52), 
        (51, 176, 203), 
        (200, 54, 131), 
        (92, 193, 61),  
        (78, 71, 183),  
        (172, 114, 82), 
        (255, 127, 14), 		# refrigerator
        (91, 163, 138), 
        (153, 98, 156), 
        (140, 153, 101),
        (158, 218, 229),		# shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  		# toilet
        (112, 128, 144),		# sink
        (96, 207, 209), 
        (227, 119, 194),		# bathtub
        (213, 92, 176), 
        (94, 106, 211), 
        (82, 84, 163),  		# otherfurn
        (100, 85, 144)
    ]