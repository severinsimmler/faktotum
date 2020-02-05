import json
import logging
from pathlib import Path
from typing import Dict, Generator, List, Tuple


class KnowledgeBase:
    def __init__(self, humans, organizations, positions):
        if positions:
            logging.info("Parsing positions...")
            self.dump = positions
            self.positions = dict(self._select_positions())
        if organizations:
            logging.info("Parsing organizations...")
            self.dump = organizations
            self.organizations = dict(self._select_organizations())
        if humans:
            logging.info("Parsing humans...")
            self.dump = humans
            self.employed_humans = dict(self._select_employed_humans())

    def export(self, directory):
        if hasattr(self, "positions"):
            with Path(directory, "filtered-positions.json").open(
                "w", encoding="utf-8"
            ) as file_:
                json.dump(self.positions, file_, ensure_ascii=False)
        if hasattr(self, "employed_humans"):
            with Path(directory, "filtered-employed-humans.json").open(
                "w", encoding="utf-8"
            ) as file_:
                json.dump(self.employed_humans, file_, ensure_ascii=False)
        if hasattr(self, "organizations"):
            with Path(directory, "filtered-organizations.json").open(
                "w", encoding="utf-8"
            ) as file_:
                json.dump(self.organizations, file_, ensure_ascii=False)

    def _select_positions(self):
        for line in self.dump:
            identifier = line["id"]
            claims = line["claims"]
            properties = dict()
            mentions = list()

            label = line["labels"].get("de")
            if label:
                mentions.append(label["value"])

            aliases = line["aliases"].get("de")
            if aliases:
                for alias in aliases:
                    mentions.append(alias["value"])

            properties["MENTIONS"] = mentions

            descriptions = line["descriptions"].get("de")
            if descriptions:
                properties["DESCRIPTION"] = descriptions["value"]

            if properties["MENTIONS"]:
                yield identifier, properties

    def _select_organizations(self):
        for line in self.dump:
            identifier = line["id"]
            claims = line["claims"]
            properties = dict()
            mentions = list()

            label = line["labels"].get("de")
            if label:
                mentions.append(label["value"])

            aliases = line["aliases"].get("de")
            if aliases:
                for alias in aliases:
                    mentions.append(alias["value"])

            properties["MENTIONS"] = mentions

            descriptions = line["descriptions"].get("de")
            if descriptions:
                properties["DESCRIPTION"] = descriptions["value"]

            # industry = list(self._format_properties(claims, "P452"))
            # if industry:
            #    properties["INDUSTRY"] = industry

            ceo = list(self._format_properties(claims, "P169"))
            if ceo:
                properties["CEO"] = ceo

            country = list(self._format_properties(claims, "P17"))
            if country:
                properties["COUNTRY"] = country

            # legal_form = list(self._format_properties(claims, "P1454"))
            # if legal_form:
            #    properties["LEGAL_FORM"] = legal_form

            if properties["MENTIONS"]:
                yield identifier, properties

    def _select_employed_humans(self):
        for line in self.dump:
            # only humans that have an employer
            if "P108" in line["claims"]:
                identifier = line["id"]
                properties = dict()
                mentions = list()

                label = line["labels"].get("de")
                if label:
                    mentions.append(label["value"])

                aliases = line["aliases"].get("de")
                if aliases:
                    for alias in aliases:
                        mentions.append(alias["value"])

                properties["MENTIONS"] = mentions

                descriptions = line["descriptions"].get("de")
                if descriptions:
                    properties["DESCRIPTION"] = descriptions["value"]

                claims = line["claims"]

                first_names = list(self._format_properties(claims, "P734"))
                if first_names:
                    properties["LAST_NAMES"] = first_names

                last_names = list(self._format_properties(claims, "P735"))
                if last_names:
                    properties["FIRST_NAMES"] = last_names

                gender = list(self._format_properties(claims, "P21"))
                if gender:
                    properties["GENDER"] = gender

                # occupation = list(self._format_properties(claims, "P106"))
                # if occupation:
                #    properties["OCCUPATION"] = occupation

                employer = list(self._format_properties(claims, "P108"))
                if employer:
                    properties["EMPLOYER"] = employer

                position = list(self._format_properties(claims, "P39"))
                if position:
                    properties["POSITION"] = position
                    relations = list(self._get_relations(claims))
                    if relations:
                        properties["RELATIONS"] = relations

                working_since = list(self._format_properties(claims, "P2031"))
                if working_since:
                    properties["WORKING_SINCE"] = working_since

                owner_of = list(self._format_properties(claims, "P1830"))
                if owner_of:
                    properties["OWNER_OF"] = owner_of

                birth_name = list(self._format_properties(claims, "P1477"))
                if birth_name:
                    properties["MENTIONS"].extend(birth_name)

                degree = list(self._format_properties(claims, "P512"))
                if degree:
                    properties["ACADEMIC_DEGREE"] = degree

                if properties["MENTIONS"]:
                    yield identifier, properties

    @staticmethod
    def _format_properties(claims, identifier):
        properties = claims.get(identifier)
        _values = set()
        if properties:
            for property_ in properties:
                if "datavalue" in property_["mainsnak"]:
                    value = property_["mainsnak"]["datavalue"]["value"]
                    if "id" in value and value["id"] not in _values:
                        _values.add(value["id"])
                        yield value["id"]
                    elif "text" in value and value["text"] not in _values:
                        _values.add(value["text"])
                        yield value["text"]

    @staticmethod
    def _get_relations(claims):
        for position in claims["P39"]:
            if "datavalue" in position["mainsnak"]:
                p = position["mainsnak"]["datavalue"]["value"]["id"]
                if "qualifiers" in position:
                    organizations = position["qualifiers"].get("P642")
                    if organizations:
                        for organization in organizations:
                            yield [p, organization["datavalue"]["value"]["id"]]
