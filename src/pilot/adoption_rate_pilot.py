from enum import Enum


BASE_CONTEXT = """
An institution providing microfinance services has started a new programme in villages across Karnataka, India.
Their services have entered your village, and as the head of the household you have been asked to consider joining the programme.
"""

DEMOGRAPHIC_ENHANCEMENT = """
Your household has the following characteristics:
- Religion: {religion}
- Caste: {caste}
- Household size: {hh_size}
- Number of rooms: {num_rooms}
- Number of beds: {num_beds}
- Access to electricity: {electricity}
- Access to a private latrine: {latrine}
- Member of a savings self-help group: {savings_group}
- Access to a bank account: {bank_account}
"""

NARRATIVE_ENHANCEMENT = """
Your household has been described as follows:
{narrative}
"""

INFORMER = "You were told about the programme by a neighbour."
INFORMER_PROFILE = """
Your neighbour has the following characteristics:
- Religion: {religion}
- Caste: {caste}
- Household size: {hh_size}
- Number of rooms: {num_rooms}
- Number of beds: {num_beds}
- Access to electricity: {electricity}
- Access to a private latrine: {latrine}
- Member of a savings self-help group: {savings_group}
- Access to a bank account: {bank_account}
- Occupation: {occupation}
"""
INFORMER_NARRATIVE = """
Your neighbour has been described as follows:
{narrative}
"""
JOINER = "They joined the programme."
NON_JOINER = "They have not joined the programme."

FORMAT_INSTRUCTION = "Does your household join the programme? Your response must have {Y} for yes or {N} for no as the last thing you write on a new line."

class LLMs(Enum):
    GPT_5_4_NANO = "gpt-5.4-nano"
    HAIKU_4_5 = "claude-haiku-4-5-20251001"
    GROK_4_2 = "grok-4.20-0309-non-reasoning"



"""
TODO: 
- modular prompt design. Always starts with the base context, and terminate with the format instruction. This constitutes the base case. For the rest of the modalities, we pick a sample adopter and a sample non-adopter household from village 6 which dont have "not known" for the required fields, and repeat the tests for both samples. Axis A: optionally add demographic (A1) OR narrative enhancement (A2). Axis B: optionally add the "informer" line, plus either the profile (B1) OR informer narrative (B2). Axis C: optionally add the "informer line" plus either the joiner (C1) OR non-joiner (C2) detail. So we have 3 LLMs x the base case + 2 samples x 3 LLMs x 27 prompt designs (3x3 axis combination x 3 axis options(not,1,2) ).
- we should save logs of the prompts we send to the LLMs, and the responses we get back, for each household in each round. save them in csv files named <llm_label>_<promptdesignlabel>.csv, where each row is a repetition and prompt, generated text response, extracted decision, token counts will be the columns.
- we repeat each configuration 10 times.
- we plot the adoption rate for each configuration with STD error bars.
- we also evaluate the statistical significance of differences between the adopter vs. non-adopter samples for each prompt design (except the base case which does not use samples). The goal with this is to see if the prompt design can be used to detect differences in adoption rates between using an adopter sample and a non-adopter sample, and to determine which (if any) prompt design is best for this.
"""