# TODO: Still need to decide final SAGAT Q list before finishing this script

def sagat_grade_oneuser():
    """
    What this does: Grades all SAGAT questions (from all 12 surveys) for ONE subject.

    Inputs:
    * 12 raw SAGAT survey excel export from Qualtrics. Must include:
        - Round number (1,2,3,4)
        - Survey number for that round (1,2,3)
    * Four jsonl files (one for each of the subject's four rounds)

    Pseudocode process:
    * Loop through all questions in the survey excel export
        - Access the appropriate .jsonl file based on the round number
        - Map the survey number to a timestamp (survey 1 = 60sec, survey2 = 120 sec, survey3 = 180 sec)
        - Grade each question
            - Map location questions: "1" if clicked location is within 250 pixels of true location
            -  Current health: "1" if within 3 HP of truth
            - Percentage: TODO
            - Others TBD TODO

    Output: A nested dictionary of grades for the subject, ala:

    grades = {'round1_Q1': 1, 'round1_Q2': 0, ...
               'round2_Q1': 0, ...
               }
    """

    pass

def sagat_grade_all_users():
    """
    Will call sagat_grade_oneuser() repeatedly to write all user data to the same file

    Inputs:
        * An excel sheet to write to (will write one line per subject)

    """
    pass