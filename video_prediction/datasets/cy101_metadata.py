import numpy as np

# metadata for the cy101 dataset, used by the CY101Dataset class

CATEGORIES = set(['basket', 'weight', 'smallstuffedanimal', 'bigstuffedanimal', 'metal', 'timber', 'pasta', 'tin', 'pvc',
              'cup', 'can', 'bottle', 'cannedfood', 'medicine', 'tupperware', 'cone', 'noodle', 'eggcoloringcup', 'egg',
              'ball'])

OBJECTS = set([
    'ball_base', 'can_coke', 'egg_rough_styrofoam', 'noodle_3', 'timber_square', 'ball_basket', 'can_red_bull_large',
    'egg_smooth_styrofoam', 'noodle_4', 'timber_squiggle', 'ball_blue', 'can_red_bull_small', 'egg_wood', 'noodle_5',
    'tin_pokemon',
    'ball_transparent', 'can_starbucks', 'eggcoloringcup_blue', 'pasta_cremette', 'tin_poker', 'ball_yellow_purple',
    'cannedfood_chili',
    'eggcoloringcup_green', 'pasta_macaroni', 'tin_snack_depot', 'basket_cylinder', 'cannedfood_cowboy_cookout',
    'eggcoloringcup_orange',
    'pasta_penne', 'tin_snowman', 'basket_funnel', 'cannedfood_soup', 'eggcoloringcup_pink', 'pasta_pipette', 'tin_tea',
    'basket_green',
    'cannedfood_tomato_paste', 'eggcoloringcup_yellow', 'pasta_rotini', 'tupperware_coffee_beans', 'basket_handle',
    'cannedfood_tomatoes',
    'medicine_ampicillin', 'pvc_1', 'tupperware_ground_coffee', 'basket_semicircle', 'cone_1', 'medicine_aspirin',
    'pvc_2', 'tupperware_marbles',
    'bigstuffedanimal_bear', 'cone_2', 'medicine_bilberry_extract', 'pvc_3', 'tupperware_pasta',
    'bigstuffedanimal_bunny', 'cone_3',
    'medicine_calcium', 'pvc_4', 'tupperware_rice', 'bigstuffedanimal_frog', 'cone_4', 'medicine_flaxseed_oil', 'pvc_5',
    'weight_1',
    'bigstuffedanimal_pink_dog', 'cone_5', 'metal_flower_cylinder', 'smallstuffedanimal_bunny', 'weight_2',
    'bigstuffedanimal_tan_dog',
    'cup_blue', 'metal_food_can', 'smallstuffedanimal_chick', 'weight_3', 'bottle_fuse', 'cup_isu',
    'metal_mix_covered_cup',
    'smallstuffedanimal_headband_bear', 'weight_4', 'bottle_google', 'cup_metal', 'metal_tea_jar',
    'smallstuffedanimal_moose',
    'weight_5', 'bottle_green', 'cup_paper_green', 'metal_thermos', 'smallstuffedanimal_otter', 'bottle_red',
    'cup_yellow', 'timber_pentagon', 'bottle_sobe', 'egg_cardboard', 'noodle_1', 'timber_rectangle', 'can_arizona',
    'egg_plastic_wrap', 'noodle_2', 'timber_semicircle'
])

SORTED_OBJECTS = sorted(list(OBJECTS))

DESCRIPTORS_BY_OBJECT = {
    "ball_base":["hard","ball","green","small","round","toy"],
    "ball_basket":["squishy","soft","brown","ball","rubber","round","toy"],
    "ball_blue":["ball","blue","plastic","hard","round","toy"],
    "ball_transparent":["ball","blue","transparent","hard","small","round","toy"],
    "ball_yellow_purple":["ball","yellow","purple","multi-colored","soft","small","round","toy"],
    "basket_cylinder":["basket","container","wicker","cylindrical","yellow","light","empty"],
    "basket_funnel":["basket","container","wicker","cylindrical","red","yellow","multi-colored","empty"],
    "basket_green":["basket","green","container","wicker","empty"],
    "basket_handle":["basket","brown","container","wicker","handle","empty"],
    "basket_semicircle":["basket","yellow","container","wicker","empty"],
    "bigstuffedanimal_bear":["squishy","stuffed animal","bear","brown","soft","big","deformable","toy"],
    "bigstuffedanimal_bunny":["squishy","stuffed animal","bunny","brown","soft","big","deformable","toy"],
    "bigstuffedanimal_frog":["squishy","stuffed animal","green","frog","soft","big","deformable","toy"],
    "bigstuffedanimal_pink_dog":["squishy","stuffed animal","pink","dog","soft","big","deformable","toy"],
    "bigstuffedanimal_tan_dog":["squishy","stuffed animal","yellow","dog","soft","big","deformable","toy"],
    "bottle_fuse":["cylindrical","bottle","plastic","empty","container","hard","light","purple"],
    "bottle_google":["cylindrical","water bottle","bottle","plastic","blue","empty","container","hard","light"],
    "bottle_green":["cylindrical","bottle","water bottle","empty","plastic","container","green","hard","light"],
    "bottle_red":["cylindrical","bottle","water bottle","empty","plastic","container","red","squishy","light"],
    "bottle_sobe":["cylindrical","bottle","purple","plastic","hard","container","empty","light","cylindrical"],
    "can_arizona":["green","cylindrical","can","metal","aluminum","large","empty","container","open","cylindrical"],
    "can_coke":["red","cylindrical","can","metal","aluminum","small","empty","container","open","cylindrical"],
    "can_red_bull_large":["blue","cylindrical","can","metal","aluminum","large","empty","container","open","cylindrical"],
    "can_red_bull_small":["blue","cylindrical","can","metal","aluminum","small","empty","container","open","cylindrical"],
    "can_starbucks":["cylindrical","can","metal","aluminum","small","empty","container","open","cylindrical"],
    "cannedfood_chili":["cylindrical","can","full","metal","multicolored"],
    "cannedfood_cowboy_cookout":["cylindrical","can","full","metal","multicolored"],
    "cannedfood_soup":["cylindrical","can","full","metal","multicolored"],
    "cannedfood_tomato_paste":["cylindrical","can","full","metal","multicolored"],
    "cannedfood_tomatoes":["cylindrical","can","full","metal","multicolored"],
    "cone_1":["cone","green","small","short","styrofoam","light"],
    "cone_2":["cone","green","small","short","styrofoam","light"],
    "cone_3":["cone","green","medium","styrofoam","light"],
    "cone_4":["cone","green","tall","styrofoam","light"],
    "cone_5":["cone","green","tall","big","styrofoam"],
    "cup_blue":["cup","blue","plastic","empty"],
    "cup_isu":["cup","red","empty","plastic"],
    "cup_metal":["cup","metal","empty"],
    "cup_paper_green":["cup","paper","green","empty"],
    "cup_yellow":["cup","yellow","plastic","empty"],
    "egg_cardboard":["egg","green","small","cardboard"],
    "egg_plastic_wrap":["egg","plastic","small","green"],
    "egg_rough_styrofoam":["egg","small","styrofoam","green"],
    "egg_smooth_styrofoam":["egg","small","styrofoam","green"],
    "egg_wood":["egg","small","wood","green"],
    "eggcoloringcup_blue":["cup","plastic","small","cylindrical","blue","empty","short","light"],
    "eggcoloringcup_green":["cup","plastic","small","cylindrical","green","empty","short","light"],
    "eggcoloringcup_orange":["cup","plastic","small","cylindrical","orange","empty","short","light"],
    "eggcoloringcup_pink":["cup","plastic","small","cylindrical","pink","empty","short","light"],
    "eggcoloringcup_yellow":["cup","plastic","small","cylindrical","yellow","empty","short","light"],
    "medicine_ampicillin":["medicine","container","full","closed","plastic","pills","hard","transparent","orange","short","small"],
    "medicine_aspirin":["medicine","container","full","closed","plastic","pills","hard","transparent","white","short", "small"],
    "medicine_bilberry_extract":["medicine","container","full","closed","plastic","pills","hard","green","short","small"],
    "medicine_calcium":["medicine","container","full","closed","plastic","pills","hard","transparent","orange","short","small"],
    "medicine_flaxseed_oil":["medicine","container","full","closed","plastic","pills","hard","yellow","short","small"],
    "metal_flower_cylinder":["metal","cylinder","tall","large","empty","container","shiny","closed"],
    "metal_food_can":["metal","cylinder","short","empty","container","shiny","closed"],
    "metal_mix_covered_cup":["metal","can","cylinder","empty","open","shiny"],
    "metal_tea_jar":["metal","can","cylinder","empty","open","shiny"],
    "metal_thermos":["metal","cylinder","bottle","empty","closed","shiny"],
    "noodle_1":["pink","foam","soft","deformable","light","short","toy"],
    "noodle_2":["pink","foam","soft","deformable","light","short","toy"],
    "noodle_3":["pink","foam","soft","deformable","toy","light"],
    "noodle_4":["pink","foam","soft","deformable","toy","tall","light"],
    "noodle_5":["pink","foam","soft","deformable","toy","tall","light","big"],
    "pasta_cremette":["green","multicolored","small","pasta","box","paper","container","full","closed","deformable","rectangular"],
    "pasta_macaroni":["blue","multicolored","pasta","box","paper","container","full","closed","deformable","rectangular"],
    "pasta_penne":["yellow","multicolored","pasta","box","paper","container","full","closed","deformable","large","rectangular"],
    "pasta_pipette":["blue","multicolored","pasta","box","paper","container","full","closed","deformable","large","rectangular"],
    "pasta_rotini":["yellow","multicolored","pasta","box","paper","container","full","closed","deformable","large","rectangular"],
    "pvc_1":["pvc","plastic","cylindrical","round","short","hard","green","pipe","small","light"],
    "pvc_2":["pvc","plastic","cylindrical","round","short","hard","green","pipe","small"],
    "pvc_3":["pvc","plastic","cylindrical","round","short","hard","green","pipe"],
    "pvc_4":["pvc","plastic","cylindrical","round","short","hard","green","pipe","wide"],
    "pvc_5":["pvc","plastic","cylindrical","round","short","hard","green","pipe","wide"],
    "smallstuffedanimal_bunny":["squishy","stuffed animal","soft","small","deformable","toy","light","pink"],
    "smallstuffedanimal_chick":["squishy","stuffed animal","soft","small","deformable","toy","light","green"],
    "smallstuffedanimal_headband_bear":["squishy","stuffed animal","soft","small","deformable","toy","light","brown"],
    "smallstuffedanimal_moose":["squishy","stuffed animal","soft","small","deformable","toy","light","brown"],
    "smallstuffedanimal_otter":["squishy","stuffed animal","soft","small","deformable","toy","light","brown"],
    "timber_pentagon":["tall","wood","brown","stick","block","hard"],
    "timber_rectangle":["tall","wood","brown","stick","block","hard"],
    "timber_semicircle":["tall","wood","brown","stick","block","hard"],
    "timber_square":["tall","wood","brown","stick","block","hard"],
    "timber_squiggle":["tall","wood","brown","stick","block","hard"],
    "tin_pokemon":["box","container","closed","metal","empty","shiny","rectangular","hard","large","tall","yellow","multi-colored"],
    "tin_poker":["box","container","closed","metal","empty","shiny","rectangular","hard","tall","blue","multicolored"],
    "tin_snack_depot":["box","container","closed","metal","empty","shiny","rectangular","hard","large","tall","brown","multicolored"],
    "tin_snowman":["box","container","closed","metal","empty","shiny","rectangular","hard","small","short","blue"],
    "tin_tea":["box","container","closed","metal","empty","shiny","rectangular","hard","small","short","brown"],
    "tupperware_coffee_beans":["red","plastic","container","closed","full","hard"],
    "tupperware_ground_coffee":["red","plastic","container","closed","full","hard"],
    "tupperware_marbles":["red","plastic","container","closed","full","hard"],
    "tupperware_pasta":["red","plastic","container","closed","full","hard"],
    "tupperware_rice":["red","plastic","container","closed","full","hard"],
    "weight_1":["blue","tall","cylindrical","empty","closed","container","plastic"],
    "weight_2":["blue","tall","cylindrical","closed","container","plastic"],
    "weight_3":["blue","tall","cylindrical","full","closed","container","plastic"],
    "weight_4":["blue","tall","cylindrical","full","closed","container","plastic"],
    "weight_5":["blue","tall","cylindrical","full","closed","container","plastic"]
}

DESCRIPTOR_CODES = {'aluminum': 0, 'ball': 1, 'basket': 2, 'bear': 3,
    'big': 4, 'block': 5, 'blue': 6, 'bottle': 7, 'box': 8, 'brown': 9,
    'bunny': 10, 'can': 11, 'cardboard': 12, 'closed': 13, 'cone': 14,
    'container': 15, 'cup': 16, 'cylinder': 17, 'cylindrical': 18, 'deformable': 19,
    'dog': 20, 'egg': 21, 'empty': 22, 'foam': 23, 'frog': 24, 'full': 25, 'green': 26,
    'handle': 27, 'hard': 28, 'large': 29, 'light': 30, 'medicine': 31, 'medium': 32,
    'metal': 33, 'multi-colored': 34, 'multicolor': 35, 'multicolored': 36, 'open': 37,
    'orange': 38, 'paper': 39, 'pasta': 40, 'pills': 41, 'pink': 42, 'pipe': 43,
    'plastic': 44, 'purple': 45, 'pvc': 46, 'rectangular': 47, 'red': 48, 'round': 49,
    'rubber': 50, 'shiny': 51, 'short': 52, 'small': 53, 'soft': 54,
    'squishy': 55, 'stick': 56, 'stuffed animal': 57, 'styrofoam': 58, 'tall': 59,
    'toy': 60, 'transparent': 61, 'water bottle': 62,
    'white': 63, 'wicker': 64, 'wide': 65, 'wood': 66, 'yellow': 67}

BEHAVIORS = ['crush', 'grasp', 'lift_slow', 'shake', 'poke', 'push', 'tap', 'low_drop', 'hold']

TRIALS = ['exec_1', 'exec_2', 'exec_3', 'exec_4', 'exec_5']

#########
# category_split
#   Splits words on objects with balanced categories to prepare for
#   5-fold cross validation. Assumes objects are in groupings/categories
#   of exactly 5 with unique prefixes.
# Returns: list of 5 lists of objects to be placed in the test set,
#   one list for each of the 5 folds of cross validation
##########
def category_split(num_folds=5):
    ## test assumptions
    if len(SORTED_OBJECTS) != 100:
        raise Exception("split is intended to work for exactly 100 objects")
    
    assert type(num_folds) == int, "The num_folds argument must be an integer number of splits."
    if num_folds != 1 and num_folds != 5:
        raise Exception("only 1 and 5 fold cross validation currently supported")

    ## semi-randomly split data
    test_objects_by_fold = [set([]) for i in range(num_folds)]
    # for each of 20 categories, each containing 5 objects
    for category_i in range(len(SORTED_OBJECTS)//5):
        low_ind = 5*category_i
        random_list = np.random.permutation(5)

        # for each of the 5 objects in that category
        for cross_val_fold_i in range(num_folds):
            object_i = low_ind + random_list[cross_val_fold_i]
            if SORTED_OBJECTS[object_i][0] != SORTED_OBJECTS[low_ind][0]:
                raise Exception("each grouping must have exactly 5 objects with identical prefix")
            else:
                test_objects_by_fold[cross_val_fold_i].add(SORTED_OBJECTS[object_i])

    return test_objects_by_fold