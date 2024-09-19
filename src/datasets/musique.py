import gzip
import json
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from datasets import load_dataset, load_from_disk
from src.configs import DataConfigs, DecoderConfigs
from src.datasets.base_dataset import BaseDataset


class MuSiQue(BaseDataset):
    """
    Closed-book MuSiQue
    """

    def __init__(
        self,
        data_configs: DataConfigs,
        **kwargs,
    ):
        super().__init__(data_configs, **kwargs)
        self.variation = data_configs.variation

        self.data_filename = os.path.join(self.data_dir, "musique_ans_v1.0_dev.jsonl")

        # Prepare data
        self.data = self.parse_data()

    def parse_data(self) -> List[dict]:
        data = []

        with open(self.data_filename, "r") as f:
            for i, line in enumerate(f):
                instance = json.loads(line)
                if instance["answerable"]:
                    sample_id = instance["id"]
                    sample_type = sample_id.split("__")[0]
                    data += [
                        {
                            "idx": sample_id,
                            "type": sample_type,
                            "paragraphs": [
                                {
                                    "title": para["title"],
                                    "paragraph_text": para["paragraph_text"],
                                }
                                for para in instance["paragraphs"]
                                if para["is_supporting"]
                            ],
                            "question": instance["question"],
                            "answers": [instance["answer"]]
                            + instance["answer_aliases"],
                        }
                    ]

        if self.num_samples > 0:
            data = data[: self.num_samples]

        return data

    def create_closed_book_demo_text(self) -> List[str]:
        # Train QA Pairs: https://github.com/StonyBrookNLP/ircot/blob/main/prompts/musique/no_context_cot_qa_flan_t5.txt
        questions, answers = [], []

        questions.append("When was Neville A. Stanton's employer founded?")
        if self.variation.startswith("cot_"):
            answers.append(
                "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. So the answer is: 1862."
            )
        elif self.variation.startswith("direct_"):
            answers.append("1862")

        questions.append(
            "What is the headquarters for the organization who sets the standards for ISO 21500?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "The standards for ISO 21500 were set by International Organization for Standardization. The International Organization for Standardization has headquarters in Geneva. So the answer is: Geneva."
            )
        elif self.variation.startswith("direct_"):
            answers.append("Geneva")

        questions.append(
            "What region of the state where Guy Shepherdson was born, contains SMA Negeri 68?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Guy Shepherdson was born in Jakarta. SMA Negeri 68 Jakarta is located in Central Jakarta. So the answer is: Central Jakarta."
            )
        elif self.variation.startswith("direct_"):
            answers.append("Central Jakarta")

        questions.append(
            "When was the first railway line constructed between Kotri and the city where Marie Adelaide Leprosy Centre is located?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Marie Adelaide Leprosy Centre is located in Karachi. The first railway line between Kotri and Karachi was constructed in April 1858. So the answer is: April 1858."
            )
        elif self.variation.startswith("direct_"):
            answers.append("April 1858")

        questions.append(
            "What county is Hebron located in, in the same province the Heritage Places Protection Act applies to?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Heritage Places Protection Act applies to the jurisdiction of Prince Edward Island. Hebron, Prince Edward Island is located in the Prince County. So the answer is: Prince County."
            )
        elif self.variation.startswith("direct_"):
            answers.append("Prince County")

        questions.append(
            "When did the first large winter carnival take place in the city where CIMI-FM is licensed to broadcast?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "CIMI-FM is licensed to broadcast in Quebec City. The first large winter carnival in Quebec City took place in 1894. So the answer is: 1894."
            )
        elif self.variation.startswith("direct_"):
            answers.append("1894")

        questions.append(
            "When did the first large winter carnival happen in Olivier Robitaille's place of birth?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Olivier Robitaille was born in Quebec City. The first large winter carnival in Quebec City happened in the 1894. So the answer is: 1894."
            )
        elif self.variation.startswith("direct_"):
            answers.append("1894")

        questions.append("When did Britain withdraw from the country containing Hoora?")
        if self.variation.startswith("cot_"):
            answers.append(
                "Hoora is in the country of Bahrain. Britain withdrew from Bahrain in 1971. So the answer is: 1971."
            )
        elif self.variation.startswith("direct_"):
            answers.append("1971")

        questions.append(
            "When did Britain withdraw from the country where the village of Wadyan is found?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Wadyan is in the country of Bahrain. Britain withdraw from Bahrain in 1971. So the answer is: 1971."
            )
        elif self.variation.startswith("direct_"):
            answers.append("1971")

        questions.append(
            "What did the publisher of Banjo-Tooie rely primarily on for its support?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "The publisher of Banjo-Tooie is Nintendo. Nintendo relied primarily for its support on first-party games. So the answer is: first-party games."
            )
        elif self.variation.startswith("direct_"):
            answers.append("first-party games")

        questions.append(
            "What shares a border with Rivière-Verte in the province WRSU-FM broadcasts in?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "WRSU-FM was licensed to broadcast to New Brunswick. Rivière-Verte, New Brunswick shares border with Edmundston. So the answer is: Edmundston."
            )
        elif self.variation.startswith("direct_"):
            answers.append("Edmundston")

        questions.append(
            "When was the state of emergency declared in the country where the Senate is located?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "The Senate is in the country of Kenya. The state of emergency was declared in Kenya on 20 October 1952. So the answer is: 20 October 1952."
            )
        elif self.variation.startswith("direct_"):
            answers.append("20 October 1952")

        # Commented out because it's weird :/ The second reasoning step is very unrelated.
        # questions.append(
        #     "Where is the crying stone found in the country in which Raphael Tuju holds citizenship?"
        # )
        # if self.variation.startswith("cot_"):
        # answers.append(
        #     "Raphael Tuju is a citizen of Kenya. The crying stone in Kenya is found along the highway towards Kisumu. So the answer is: along the highway towards Kisumu."
        # )
        # elif self.variation.startswith("direct_"):
        #     answers.append("along the highway towards Kisumu")

        questions.append(
            "Where does the Snake River start, in the state where Lima Mountain is located?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Lima Mountain is located in the state of Minnesota. The snake river in Minnesota starts in southern Aitkin County. So the answer is: southern Aitkin County."
            )
        elif self.variation.startswith("direct_"):
            answers.append("southern Aitkin County")

        # Commented out because it's weird :/ The question is not answerable upon checking the evidence
        # questions.append(
        #     "What genre is the record label of the performer of So Long, See You Tomorrow associated with?"
        # )
        # if self.variation.startswith("cot_"):
        # answers.append(
        #     "The performer of So Long, See You Tomorrow is Bombay Bicycle Club. The record label of Bombay Bicycle Club is Island Records. The genre of Island Records is jazz. So the answer is: jazz."
        # )
        # elif self.variation.startswith("direct_"):
        #     answers.append("jazz")

        questions.append(
            "In which county was the birthplace of the Smoke in tha City performer?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "The performer of Smoke in tha City is MC Eiht. MC Eiht's birthplace is Compton. Compton is located in the county of Los Angeles County. So the answer is: Los Angeles County."
            )
        elif self.variation.startswith("direct_"):
            answers.append("Los Angeles County")

        # Commented out because it's weird :/ The question is not answerable upon checking the evidence
        # questions.append(
        #     "What is the genre of the record label of the band that performed on the Crush Tour?"
        # )
        # if self.variation.startswith("cot_"):
        # answers.append(
        #     "The Crush Tour is performed by the band Bon Jovi. The record label of Bon Jovi is Island Records. The genre of Island Records is jazz. So the answer is: jazz."
        # )
        # elif self.variation.startswith("direct_"):
        #     answers.append("jazz")

        questions.append(
            "How long is the US border with the country that borders the state where Finding Dory takes place?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Finding Dory is supposed to take place in California. The country that shares a border with California is Mexico. The length of the us border with Mexico is 1,989 mi. So the answer is: 1,989 mi."
            )
        elif self.variation.startswith("direct_"):
            answers.append("1,989 mi")

        questions.append(
            "What weekly publication in the Connecticut city with the most Zagat rated restaurants is issued by university of America-Lite: How Imperial Academia Dismantled Our Culture's author?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "The author of America-Lite: How Imperial Academia Dismantled Our Culture is David Gelernter. David Gelernter was educated at the Yale University. The city in Connecticut that has the highest number of Zagat-rated restaurants is New Haven. The weekly publication in New Haven that is issued by Yale University is Yale Herald. So the answer is: Yale Herald."
            )
        elif self.variation.startswith("direct_"):
            answers.append("Yale Herald")

        questions.append(
            "How many countries in Pacific National University's continent are recognized by the organization that mediated the truce ending the Iran-Iraq war?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Pacific National University is located in Khabarovsk, Russia Khabarovsk, Russian is in the continent of Asia. The entity that mediated the truce which ended the Iran-Iraq War is the UN. The number of member states that UN recognises in Asia is 53. So the answer is: 53."
            )
        elif self.variation.startswith("direct_"):
            answers.append("53")

        demo_texts = []
        if self.kwargs["use_chat_template"]:
            for i in range(len(questions)):
                demo_texts += [
                    f"Q: {questions[i]}\nA:",
                    answers[i],
                ]
        else:
            for i in range(len(questions)):
                demo_texts += [f"Q: {questions[i]}\nA: {answers[i]}"]
        return demo_texts

    @staticmethod
    def _prepare_contexts(contexts) -> str:
        supporting_contexts = [
            f"Wikipedia Title: {context['title']}\n{context['paragraph_text']} "
            for context in contexts
        ]
        return "\n\n".join(supporting_contexts)

    def create_open_book_demo_text(self) -> List[str]:
        # Train QA Pairs: https://github.com/StonyBrookNLP/ircot/blob/main/prompts/musique/no_context_cot_qa_flan_t5.txt
        contexts, questions, answers = [], [], []

        contexts.append(
            self._prepare_contexts(
                [
                    {
                        "title": "Neville A. Stanton",
                        "paragraph_text": 'Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.',
                    },
                    {
                        "title": "Southampton",
                        "paragraph_text": "The University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.",
                    },
                ]
            )
        )
        questions.append("When was Neville A. Stanton's employer founded?")
        if self.variation.startswith("cot_"):
            answers.append(
                "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. So the answer is: 1862."
            )
        elif self.variation.startswith("direct_"):
            answers.append("1862")

        contexts.append(
            [
                {
                    "title": "ISO 21500",
                    "paragraph_text": "ISO 21500:2012, Guidance on Project Management, is an international standard developed by the International Organization for Standardization, or ISO starting in 2007 and released in 2012. It was intended to provide generic guidance, explain core principles and what constitutes good practice in project management. The ISO technical committee dealing with project management, ISO/PC 236 was held by the American National Standards Institute (ANSI) which had approved four standards that used PMI materials. one of which was ANSI/PMI 99-001-2008, A Guide to the Project Management Body of Knowledge - 4th Edition (PMI BoK® Guide - 4th Edition) (revision and re-designation of ANSI/PMI 99-001-2004): 11/20/2008.",
                },
                {
                    "title": "ISO/TC 68",
                    "paragraph_text": "ISO/TC 68 is a technical committee formed within the International Organization for Standardization (ISO), of Geneva, Switzerland, tasked with developing and maintaining international standards covering the areas of banking, securities, and other financial services. As the standards organization under ISO responsible for the development of all international financial services standards, ISO/TC 68 plays a key role in the development and adoption of new technologies in the banking, brokerage and insurance industries. Many of its current work projects involve developing ecommerce standards such as better online security for financial transactions, XML standards for financial transactions and standards to reduce the cost and delays of international financial transactions. The membership of ISO/TC 68, consists of more than 30 organizations assigned by participating national standards bodies plus additional international standards development organizations that work collaboratively toward global financial services standards development.",
                },
            ]
        )
        questions.append(
            "What is the headquarters for the organization who sets the standards for ISO 21500?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "The standards for ISO 21500 were set by International Organization for Standardization. The International Organization for Standardization has headquarters in Geneva. So the answer is: Geneva."
            )
        elif self.variation.startswith("direct_"):
            answers.append("Geneva")

        contexts.append(
            [
                {
                    "title": "SMA Negeri 68 Jakarta",
                    "paragraph_text": "SMA Negeri 68 Jakarta (SMANED) is a public high school located at Salemba Raya street in Central Jakarta, Indonesia. The school is in one complex with SMP Negeri 216 Jakarta, SD Negeri Kenari, and Menza functional building. It was established on August 29, 1981 after being inaugurated by President Soeharto. In 2006, it was appointed to become RSBI (Rintisan Sekolah Bertaraf Internasional). Today, there are 840 students and 103 teachers and staff.",
                },
                {
                    "title": "Guy Shepherdson",
                    "paragraph_text": "Guy Shepherdson (born 17 February 1982 in Jakarta, Indonesia) is an Australian former rugby union professional footballer. He played as a tight-head prop for the Brumbies and Reds in the Super Rugby competition and played for the Australian national team, the Wallabies.",
                },
            ]
        )
        questions.append(
            "What region of the state where Guy Shepherdson was born, contains SMA Negeri 68?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Guy Shepherdson was born in Jakarta. SMA Negeri 68 Jakarta is located in Central Jakarta. So the answer is: Central Jakarta."
            )
        elif self.variation.startswith("direct_"):
            answers.append("Central Jakarta")

        contexts.append(
            [
                {
                    "title": "Marie Adelaide Leprosy Centre",
                    "paragraph_text": "Marie Adelaide Leprosy Centre (MALC) in Karachi, Pakistan was run by Dr. Ruth Pfau, who was also a Roman Catholic religious sister of the Society of Daughters of the Heart of Mary, originally of German descent. Its social work department was founded 1962 by Dr. I. K. Gill and work for the leprosy patients and their family members was started. A Leprosy Clinic was bought in April 1963 and patients from all over Pakistan and even from Afghanistan came for treatment.",
                },
                {
                    "title": "Kotri Junction railway station",
                    "paragraph_text": "Kotri Junction station is among the oldest railway stations in Pakistan. It served as the northern terminus point of the Scinde Railway, which was established in March 1855. A railway line was to be constructed between Karachi and Kotri and work on the Karachi terminus commenced in April 1858. By 13 May 1861, the station opened to the public. This was the first railway line for public traffic between Karachi and Kotri, a distance of 108 miles (174 km).",
                },
            ]
        )
        questions.append(
            "When was the first railway line constructed between Kotri and the city where Marie Adelaide Leprosy Centre is located?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Marie Adelaide Leprosy Centre is located in Karachi. The first railway line between Kotri and Karachi was constructed in April 1858. So the answer is: April 1858."
            )
        elif self.variation.startswith("direct_"):
            answers.append("April 1858")

        contexts.append(
            [
                {
                    "title": "Heritage Places Protection Act",
                    "paragraph_text": "The Heritage Places Protection Act is a provincial statute which allows for the recognition and protection of cultural heritage and natural heritage properties in the province of Prince Edward Island, Canada.",
                },
                {
                    "title": "Hebron, Prince Edward Island",
                    "paragraph_text": "Hebron is a Canadian rural community in Prince County, Prince Edward Island. It is located in the township of Lot 8, Prince Edward Island, south of O'Leary.",
                },
            ]
        )
        questions.append(
            "What county is Hebron located in, in the same province the Heritage Places Protection Act applies to?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Heritage Places Protection Act applies to the jurisdiction of Prince Edward Island. Hebron, Prince Edward Island is located in the Prince County. So the answer is: Prince County."
            )
        elif self.variation.startswith("direct_"):
            answers.append("Prince County")

        contexts.append(
            [
                {
                    "title": "Quebec Winter Carnival",
                    "paragraph_text": "The Quebec Winter Carnival (French: Carnaval de Québec), commonly known in both English and French as Carnaval, is a pre-Lenten festival held in Quebec City. After being held intermittently since 1894, the Carnaval de Québec has been celebrated annually since 1955. That year Bonhomme Carnaval, the mascot of the festival, made his first appearance. Up to one million people attended the Carnaval de Québec in 2006 making it the largest winter festival in the world.",
                },
                {
                    "title": "CIMI-FM",
                    "paragraph_text": "CIMI-FM was a French-language talk radio and modern rock radio station in Quebec City, Quebec, Canada. The station broadcast at 103.7 FM and broadcasts from the borough of Charlesbourg.",
                },
            ]
        )
        questions.append(
            "When did the first large winter carnival take place in the city where CIMI-FM is licensed to broadcast?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "CIMI-FM is licensed to broadcast in Quebec City. The first large winter carnival in Quebec City took place in 1894. So the answer is: 1894."
            )
        elif self.variation.startswith("direct_"):
            answers.append("1894")

        contexts.append(
            [
                {
                    "title": "Olivier Robitaille",
                    "paragraph_text": "He was born in Quebec City in 1811, the son of carpenter Étienne Robitaille and his wife Marie Moisan. He studied at the Petit Séminaire de Québec and later trained in medicine with Joseph Morrin, working as an intern at the Marine and Emigrant Hospital. He received his doctorate degree in medicine (MD) from Harvard University in 1838, was qualified to practice in Lower Canada later that year and set up practice in Quebec City. He also served as visiting physician for the Marine and Emigrant Hospital, serving on its board, and was physician for the Quebec jail.",
                },
                {
                    "title": "Quebec Winter Carnival",
                    "paragraph_text": "The Quebec Winter Carnival (French: Carnaval de Québec), commonly known in both English and French as Carnaval, is a pre-Lenten festival held in Quebec City. After being held intermittently since 1894, the Carnaval de Québec has been celebrated annually since 1955. That year Bonhomme Carnaval, the mascot of the festival, made his first appearance. Up to one million people attended the Carnaval de Québec in 2006 making it the largest winter festival in the world.",
                },
            ]
        )
        questions.append(
            "When did the first large winter carnival happen in Olivier Robitaille's place of birth?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Olivier Robitaille was born in Quebec City. The first large winter carnival in Quebec City happened in the 1894. So the answer is: 1894."
            )
        elif self.variation.startswith("direct_"):
            answers.append("1894")

        contexts.append(
            [
                {
                    "title": "Hoora",
                    "paragraph_text": "Along with the Central Business District, Adliya, and Juffair, Hoora is considered as one of Manama's nightlife centres, with many bars, hotels, restaurants, pubs and nightclubs (both Arabic and Western), and it is very popular with Arab visitors to Bahrain.",
                },
                {
                    "title": "British Empire",
                    "paragraph_text": "While the Suez Crisis caused British power in the Middle East to weaken, it did not collapse. Britain again deployed its armed forces to the region, intervening in Oman (1957), Jordan (1958) and Kuwait (1961), though on these occasions with American approval, as the new Prime Minister Harold Macmillan's foreign policy was to remain firmly aligned with the United States. Britain maintained a military presence in the Middle East for another decade. In January 1968, a few weeks after the devaluation of the pound, Prime Minister Harold Wilson and his Defence Secretary Denis Healey announced that British troops would be withdrawn from major military bases East of Suez, which included the ones in the Middle East, and primarily from Malaysia and Singapore. The British withdrew from Aden in 1967, Bahrain in 1971, and Maldives in 1976.",
                },
            ]
        )
        questions.append("When did Britain withdraw from the country containing Hoora?")
        if self.variation.startswith("cot_"):
            answers.append(
                "Hoora is in the country of Bahrain. Britain withdrew from Bahrain in 1971. So the answer is: 1971."
            )
        elif self.variation.startswith("direct_"):
            answers.append("1971")

        contexts.append(
            [
                {
                    "title": "Wadyan",
                    "paragraph_text": "Wadyan (Arabic: واديان) is a village in the island of Sitra, Bahrain. A branch of the National Bank of Bahrain and the Sitra police station are located in Wadyan.",
                },
                {
                    "title": "British Empire",
                    "paragraph_text": "While the Suez Crisis caused British power in the Middle East to weaken, it did not collapse. Britain again deployed its armed forces to the region, intervening in Oman (1957), Jordan (1958) and Kuwait (1961), though on these occasions with American approval, as the new Prime Minister Harold Macmillan's foreign policy was to remain firmly aligned with the United States. Britain maintained a military presence in the Middle East for another decade. In January 1968, a few weeks after the devaluation of the pound, Prime Minister Harold Wilson and his Defence Secretary Denis Healey announced that British troops would be withdrawn from major military bases East of Suez, which included the ones in the Middle East, and primarily from Malaysia and Singapore. The British withdrew from Aden in 1967, Bahrain in 1971, and Maldives in 1976.",
                },
            ]
        )
        questions.append(
            "When did Britain withdraw from the country where the village of Wadyan is found?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Wadyan is in the country of Bahrain. Britain withdraw from Bahrain in 1971. So the answer is: 1971."
            )
        elif self.variation.startswith("direct_"):
            answers.append("1971")

        contexts.append(
            [
                {
                    "title": "Banjo-Tooie",
                    "paragraph_text": 'Banjo-Tooie is a platform video game developed by Rare and originally released for the Nintendo 64 console in 2000. It is the second game in the "Banjo-Kazooie" series and the sequel to "Banjo-Kazooie". The game follows series protagonists Banjo and Kazooie as they attempt to stop the plans of the witch Gruntilda and her two sisters, who intend to vapourise the inhabitants of the game\'s world. The game features worlds that are significantly larger than those of its predecessor, requiring the player to complete challenges such as solving puzzles, jumping over obstacles, collecting items, and defeating bosses. It also includes a multiplayer mode where up to four players can compete in several minigames.',
                },
                {
                    "title": "Nintendo Entertainment System",
                    "paragraph_text": "In the longer run, however, with the NES near its end of its life many third-party publishers such as Electronic Arts supported upstart competing consoles with less strict licensing terms such as the Sega Genesis and then the PlayStation, which eroded and then took over Nintendo's dominance in the home console market, respectively. Consoles from Nintendo's rivals in the post-SNES era had always enjoyed much stronger third-party support than Nintendo, which relied more heavily on first-party games.",
                },
            ]
        )
        questions.append(
            "What did the publisher of Banjo-Tooie rely primarily on for its support?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "The publisher of Banjo-Tooie is Nintendo. Nintendo relied primarily for its support on first-party games. So the answer is: first-party games."
            )
        elif self.variation.startswith("direct_"):
            answers.append("first-party games")

        contexts.append(
            [
                {
                    "title": "Rivière-Verte, New Brunswick",
                    "paragraph_text": 'It is located 15 kilometres southeast of Edmundston along the Saint John River and the Riviere Verte. Its name translates to "Green River".',
                },
                {
                    "title": "WRSU-FM",
                    "paragraph_text": "WRSU (88.7 FM) is a non-commercial college radio station serving the greater Central New Jersey area, broadcasting from the campus of Rutgers University in New Brunswick, New Jersey. It is a student and faculty-run radio station with Rutgers faculty member Mike Pavlichko serving as its Broadcast Administrator. WRSU broadcasts on FM and streams all of its programming online.",
                },
            ]
        )
        questions.append(
            "What shares a border with Rivière-Verte in the province WRSU-FM broadcasts in?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "WRSU-FM was licensed to broadcast to New Brunswick. Rivière-Verte, New Brunswick shares border with Edmundston. So the answer is: Edmundston."
            )
        elif self.variation.startswith("direct_"):
            answers.append("Edmundston")

        contexts.append(
            [
                {
                    "title": "Mau Mau Uprising",
                    "paragraph_text": "On 20 October 1952, Governor Baring signed an order declaring a State of Emergency. Early the next morning, Operation Jock Scott was launched: the British carried out a mass - arrest of Jomo Kenyatta and 180 other alleged Mau Mau leaders within Nairobi. Jock Scott did not decapitate the movement's leadership as hoped, since news of the impending operation was leaked. Thus, while the moderates on the wanted list awaited capture, the real militants, such as Dedan Kimathi and Stanley Mathenge (both later principal leaders of Mau Mau's forest armies), fled to the forests.",
                },
                {
                    "title": "Senate of Kenya",
                    "paragraph_text": "The Senate is the upper house of the Parliament of Kenya. The Senate was first established as part of Kenya's 1963 Constitution. After being abolished in 1966, the Senate was re-established by the 2010 Constitution.",
                },
            ]
        )
        questions.append(
            "When was the state of emergency declared in the country where the Senate is located?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "The Senate is in the country of Kenya. The state of emergency was declared in Kenya on 20 October 1952. So the answer is: 20 October 1952."
            )
        elif self.variation.startswith("direct_"):
            answers.append("20 October 1952")

        # Commented out because it's weird :/ The second reasoning step is very unrelated.
        # contexts.append(
        #     [
        #         {"title": "Kakamega", "paragraph_text": "Kakamega Forest is the main tourist destination in the area. Another attraction is the Crying Stone of Ilesi located along the highway towards Kisumu. It is a 40 metres high rock dome resembling a human figure whose ``eyes ''drop water."},
        #         {"title": "Raphael Tuju", "paragraph_text": "Raphael Tuju, EGH (born 30 March 1959) is a Kenyan politician. In 2002—after a career as a journalist, TV producer, and real estate investor—Tuju was elected to parliament and has served the Government of Kenya in various capacities since that time."},
        #     ]
        # )
        # questions.append(
        #     "Where is the crying stone found in the country in which Raphael Tuju holds citizenship?"
        # )
        # if self.variation.startswith("cot_"):
        # answers.append(
        #     "Raphael Tuju is a citizen of Kenya. The crying stone in Kenya is found along the highway towards Kisumu. So the answer is: along the highway towards Kisumu."
        # )
        # elif self.variation.startswith("direct_"):
        #     answers.append("along the highway towards Kisumu")

        contexts.append(
            [
                {
                    "title": "Snake River (St. Croix River tributary)",
                    "paragraph_text": "The Snake River with its tributaries drains a 1,009 square miles (2,610 km) area of Aitkin, Kanabec, Mille Lacs and Pine counties. After initially flowing southward from its headwaters in southern Aitkin County, the Snake flows through Kanabec County, turning eastward near Mora, Minnesota, following a minor fault line. It drains into the St. Croix River 13 miles (21 km) east of Pine City, Minnesota.",
                },
                {
                    "title": "Lima Mountain",
                    "paragraph_text": "Lima Mountain is a 2238 foot summit in Cook County, Minnesota. It is located in the Lima Mountain Unit, a 2540 acre inventoried roadless area adjacent to the Boundary Waters Canoe Area. There is a 1 mile trail to the summit, where a fire tower once stood. Lima Mountain has a 328 foot rise over the saddle connecting it with the Misquah Hills High Point and Peak 2266. A trail to the summit begins along the Lima Grade (Forest Route 315) just north of its junction with Lima Mountain Road (Forest Route 152)",
                },
            ]
        )
        questions.append(
            "Where does the Snake River start, in the state where Lima Mountain is located?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Lima Mountain is located in the state of Minnesota. The snake river in Minnesota starts in southern Aitkin County. So the answer is: southern Aitkin County."
            )
        elif self.variation.startswith("direct_"):
            answers.append("southern Aitkin County")

        # Commented out because it's weird :/ The question is not answerable upon checking the evidence
        # contexts.append(
        #     [
        #         {"title": "The Antidote (Ronny Jordan album)", "paragraph_text": "The Antidote is the debut album by English jazz guitarist Ronny Jordan, that was released by Island Records in 1992."},
        #         {"title": "So Long, See You Tomorrow (album)", "paragraph_text": "So Long, See You Tomorrow is the fourth album by the London indie rock band Bombay Bicycle Club, released on 3 February 2014. The album is named after the novel of the same name by William Maxwell."},
        #         {"title": "Flaws (album)", "paragraph_text": "Flaws is the second studio album by the British indie rock band Bombay Bicycle Club, released on 9 July 2010 by Island Records. Unlike the band's previous releases, the album is entirely acoustic music, consisting of versions of their own tracks as well as cover versions of other artists. The album was produced in part by the guitarist Jamie MacColl's father, Neil MacColl, with recording taking place in February 2009 at The Church in Crouch End, London. The band started work on the album after completing their first album, \"I Had the Blues But I Shook Them Loose\"."},
        #     ]
        # )
        # questions.append(
        #     "What genre is the record label of the performer of So Long, See You Tomorrow associated with?"
        # )
        # if self.variation.startswith("cot_"):
        # answers.append(
        #     "The performer of So Long, See You Tomorrow is Bombay Bicycle Club. The record label of Bombay Bicycle Club is Island Records. The genre of Island Records is jazz. So the answer is: jazz."
        # )
        # elif self.variation.startswith("direct_"):
        #     answers.append("jazz")

        contexts.append(
            [
                {
                    "title": "Compton, California",
                    "paragraph_text": 'Compton is a city in southern Los Angeles County, California, United States, situated south of downtown Los Angeles. Compton is one of the oldest cities in the county and on May 11, 1888, was the eighth city to incorporate. As of the 2010 United States Census, the city had a total population of 96,456. It is known as the "Hub City" due to its geographic centrality in Los Angeles County. Neighborhoods in Compton include Sunny Cove, Leland, Downtown Compton, and Richland Farms. The city is generally a working class city with some middle-class neighborhoods, and is home to a relatively young population, at an average 25 years of age, compared to the American median age of 38 (based on 2018 data).',
                },
                {
                    "title": "MC Eiht",
                    "paragraph_text": 'Aaron Tyler (born May 22, 1971), better known by his stage name MC Eiht, is an American rapper and actor. Many of his songs are based on his life in Compton. His stage name was partly inspired by the numeral in KRS-One\'s name. He chose Eiht for its links to "hood culture", including Olde English 800 (8 Ball) and .38 caliber firearms. He is the "de facto" leader of West Coast hip hop group Compton\'s Most Wanted, which also included fellow Compton-based rappers Boom Bam, Tha Chill, DJ Mike T, DJ Slip and Ant Capone. He is also known for his role as A-Wax in the 1993 film "Menace II Society".',
                },
                {
                    "title": "Smoke in tha City",
                    "paragraph_text": "Smoke in tha City is the ninth studio album by American rapper MC Eiht, released September 14, 2004 on Factor House Records. It was produced by Black C-Zer and Quincy Miller. The album featuring guest performances by West Coast heavy-weights: RBX, Spice 1, Kokane, Jayo Felony and Daz Dillinger.",
                },
            ]
        )
        questions.append(
            "In which county was the birthplace of the Smoke in tha City performer?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "The performer of Smoke in tha City is MC Eiht. MC Eiht's birthplace is Compton. Compton is located in the county of Los Angeles County. So the answer is: Los Angeles County."
            )
        elif self.variation.startswith("direct_"):
            answers.append("Los Angeles County")

        # Commented out because it's weird :/ The question is not answerable upon checking the evidence
        # contexts.append(
        #     [
        #         {"title": "Bounce (Bon Jovi album)", "paragraph_text": "Bounce is the eighth studio album by American rock band Bon Jovi, released on October 8, 2002 through Island Records. Produced by Luke Ebbin, Jon Bon Jovi and Richie Sambora, the album was recorded at Sanctuary II Studio in New Jersey."},
        #         {"title": "The Crush Tour (album)", "paragraph_text": "The Crush Tour is a third concert video by American band Bon Jovi from the European leg of their Crush Tour. It was recorded on August 30, 2000 at Zurich, Switzerland. It was directed by Anthony Bongiovi. It was released on DVD in 2001."},
        #         {"title": "The Antidote (Ronny Jordan album)", "paragraph_text": "The Antidote is the debut album by English jazz guitarist Ronny Jordan, that was released by Island Records in 1992."},
        #     ]
        # )
        # questions.append(
        #     "What is the genre of the record label of the band that performed on the Crush Tour?"
        # )
        # if self.variation.startswith("cot_"):
        # answers.append(
        #     "The Crush Tour is performed by the band Bon Jovi. The record label of Bon Jovi is Island Records. The genre of Island Records is jazz. So the answer is: jazz."
        # )
        # elif self.variation.startswith("direct_"):
        #     answers.append("jazz")

        contexts.append(
            [
                {
                    "title": "Finding Dory",
                    "paragraph_text": "One year later, Dory is living with Marlin and Nemo on their reef. One day, Dory has a flashback and remembers that she has parents. She decides to look for them, but her memory problem is an obstacle. She eventually remembers that they lived at the Jewel of Morro Bay across the ocean in California, thanks to Nemo mentioning its name.",
                },
                {
                    "title": "Mexico–United States border",
                    "paragraph_text": "The total length of the continental border is 3,201 kilometers (1,989 mi). From the Gulf of Mexico, it follows the course of the Rio Grande (Río Bravo del Norte) to the border crossing at Ciudad Juárez, Chihuahua and El Paso, Texas. Westward from El Paso -- Juárez, it crosses vast tracts of the Chihuahuan and Sonoran deserts to the Colorado River Delta and San Diego -- Tijuana, before reaching the Pacific Ocean.",
                },
                {
                    "title": "Mexico–United States border",
                    "paragraph_text": "Among the U.S. states, Texas has the longest stretch of the border with Mexico, while California has the shortest. Among the states in Mexico, Chihuahua has the longest border with the United States, while Nuevo León has the shortest.",
                },
            ]
        )
        questions.append(
            "How long is the US border with the country that borders the state where Finding Dory takes place?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Finding Dory is supposed to take place in California. The country that shares a border with California is Mexico. The length of the us border with Mexico is 1,989 mi. So the answer is: 1,989 mi."
            )
        elif self.variation.startswith("direct_"):
            answers.append("1,989 mi")

        contexts.append(
            [
                {
                    "title": "New Haven, Connecticut",
                    "paragraph_text": 'New Haven is served by the daily New Haven Register, the weekly "alternative" New Haven Advocate (which is run by Tribune, the corporation owning the Hartford Courant), the online daily New Haven Independent, and the monthly Grand News Community Newspaper. Downtown New Haven is covered by an in-depth civic news forum, Design New Haven. The Register also backs PLAY magazine, a weekly entertainment publication. The city is also served by several student-run papers, including the Yale Daily News, the weekly Yale Herald and a humor tabloid, Rumpus Magazine. WTNH Channel 8, the ABC affiliate for Connecticut, WCTX Channel 59, the MyNetworkTV affiliate for the state, and Connecticut Public Television station WEDY channel 65, a PBS affiliate, broadcast from New Haven. All New York City news and sports team stations broadcast to New Haven County.',
                },
                {
                    "title": "David Gelernter",
                    "paragraph_text": 'David Hillel Gelernter (born March 5, 1955) is an American artist, writer, and professor of computer science at Yale University. He is a former national fellow at the American Enterprise Institute and senior fellow in Jewish thought at the Shalem Center, and sat on the National Endowment for the Arts. He publishes widely; his work has appeared in "The Wall Street Journal", "New York Post", "Los Angeles Times", "The Weekly Standard", "Frankfurter Allgemeine Zeitung", and elsewhere. His paintings have been exhibited in New Haven and Manhattan.',
                },
                {
                    "title": "America-Lite",
                    "paragraph_text": "America-Lite: How Imperial Academia Dismantled Our Culture (and Ushered in the Obamacrats) is a 2012 book by David Gelernter, published by Encounter Books.",
                },
                {
                    "title": "New Haven, Connecticut",
                    "paragraph_text": "Livability.com named New Haven as the Best Foodie City in the country in 2014. There are 56 Zagat-rated restaurants in New Haven, the most in Connecticut and the third most in New England (after Boston and Cambridge). More than 120 restaurants are located within two blocks of the New Haven Green. The city is home to an eclectic mix of ethnic restaurants and small markets specializing in various foreign foods. Represented cuisines include Malaysian, Ethiopian, Spanish, Belgian, French, Greek, Latin American, Mexican, Italian, Thai, Chinese, Japanese, Vietnamese, Korean, Indian, Jamaican, Cuban, Peruvian, Syrian/Lebanese, and Turkish.",
                },
            ]
        )
        questions.append(
            "What weekly publication in the Connecticut city with the most Zagat rated restaurants is issued by university of America-Lite: How Imperial Academia Dismantled Our Culture's author?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "The author of America-Lite: How Imperial Academia Dismantled Our Culture is David Gelernter. David Gelernter was educated at the Yale University. The city in Connecticut that has the highest number of Zagat-rated restaurants is New Haven. The weekly publication in New Haven that is issued by Yale University is Yale Herald. So the answer is: Yale Herald."
            )
        elif self.variation.startswith("direct_"):
            answers.append("Yale Herald")

        contexts.append(
            [
                {
                    "title": "Khabarovsk",
                    "paragraph_text": "Khabarovsk is served by the Khabarovsk Novy Airport with international flights to East Asia, Southeast Asia, European Russia, and Central Asia.",
                },
                {
                    "title": "Pacific National University",
                    "paragraph_text": "Pacific National University (PNU) is one of the largest universities in Khabarovsk Russia. It was established in 1958. Today the university trains over 21,000 in 54 majors.",
                },
                {
                    "title": "Iran",
                    "paragraph_text": "On September 22, 1980, the Iraqi army invaded the Iranian Khuzestan, and the Iran–Iraq War began. Although the forces of Saddam Hussein made several early advances, by mid 1982, the Iranian forces successfully managed to drive the Iraqi army back into Iraq. In July 1982, with Iraq thrown on the defensive, Iran took the decision to invade Iraq and conducted countless offensives in a bid to conquer Iraqi territory and capture cities, such as Basra. The war continued until 1988, when the Iraqi army defeated the Iranian forces inside Iraq and pushed the remaining Iranian troops back across the border. Subsequently, Khomeini accepted a truce mediated by the UN. The total Iranian casualties in the war were estimated to be 123,220–160,000 KIA, 60,711 MIA, and 11,000–16,000 civilians killed.",
                },
                {
                    "title": "United Nations Regional Groups",
                    "paragraph_text": "the African Group, with 54 member states the Asia - Pacific Group, with 53 member states the Eastern European Group, with 23 member states the Latin American and Caribbean Group (GRULAC), with 33 member states the Western European and Others Group (WEOG), with 28 member states, plus 1 member state (the United States) as an observer state.",
                },
            ]
        )
        questions.append(
            "How many countries in Pacific National University's continent are recognized by the organization that mediated the truce ending the Iran-Iraq war?"
        )
        if self.variation.startswith("cot_"):
            answers.append(
                "Pacific National University is located in Khabarovsk, Russia Khabarovsk, Russian is in the continent of Asia. The entity that mediated the truce which ended the Iran-Iraq War is the UN. The number of member states that UN recognises in Asia is 53. So the answer is: 53."
            )
        elif self.variation.startswith("direct_"):
            answers.append("53")

        demo_texts = []
        if self.kwargs["use_chat_template"]:
            for i in range(len(questions)):
                demo_texts += [
                    f"Q: {questions[i]}\nA:",
                    answers[i],
                ]
        else:
            for i in range(len(questions)):
                demo_texts += [f"Q: {questions[i]}\nA: {answers[i]}"]
        return demo_texts

    def build_prompt(self, contexts, question):
        if self.variation.startswith("cot_"):
            instruction = ["Answer the following question by reasoning step-by-step."]
        elif self.variation.startswith("direct_"):
            instruction = ["Answer the following question."]

        if self.variation.endswith("closed_book"):
            icl_demo = self.create_closed_book_demo_text()
            verbalised_contexts = ""
        elif self.variation.endswith("open_book"):
            icl_demo = self.create_open_book_demo_text()

            verbalised_contexts = self._prepare_contexts(contexts)
            if verbalised_contexts:
                verbalised_contexts += "\n\n"

        verbalised_question = f"Q: {question}\n"
        answer_prefix = "Answer:"
        if self.kwargs["use_chat_template"]:
            input_text_prompt = [
                instruction
                + icl_demo
                + [f"{verbalised_contexts}{verbalised_question}{answer_prefix}"]
            ]
            prompted_question_wo_context = [
                instruction + [f"{verbalised_question}{answer_prefix}"]
            ]
        else:
            instruction = instruction[0]
            icl_demo = "\n\n".join(icl_demo) + "\n\n"
            input_text_prompt = (
                instruction
                + icl_demo
                + (f"{verbalised_contexts}{verbalised_question}{answer_prefix}")
            )
            prompted_question_wo_context = (
                instruction + icl_demo + f"{verbalised_question}{answer_prefix}"
            )
        return {
            "verbalised_instruction": instruction,
            "verbalised_icl_demo": icl_demo,
            "verbalised_contexts": verbalised_contexts,
            "verbalised_question": verbalised_question,
            "verbalised_answer_prefix": answer_prefix,
            "prompted_question": input_text_prompt,
            "prompted_question_wo_context": prompted_question_wo_context,
        }

    def __getitem__(self, idx):
        sample = self.data[idx]

        # For attention analysis
        prompt = self.build_prompt(sample["paragraphs"], sample["question"])

        sample["verbalised_instruction"] = prompt["verbalised_instruction"]
        sample["verbalised_icl_demo"] = prompt["verbalised_icl_demo"]
        sample["verbalised_contexts"] = prompt["verbalised_contexts"]
        sample["verbalised_question"] = prompt["verbalised_question"]
        sample["verbalised_answer_prefix"] = prompt["verbalised_answer_prefix"]

        sample["prompted_question"] = prompt["prompted_question"]
        sample["prompted_question_wo_context"] = prompt["prompted_question_wo_context"]

        return sample

    def __len__(self):
        return len(self.data)
