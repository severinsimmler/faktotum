from typing import Dict, Generator, Tuple


class KnowledgeBase:
    def __init__(self, humans, organizations, positions):
        self.positions = dict(self._select_positions(positions))
        self.employed_humans = dict(self._select_employed_humans(humans))
        self.organizations = dict(self._select_organizations(organizations))

    def _update_human_properties(self):
        for identifier, properties in self.employed_humans.items():
            

    def _select_positions(dump):
        for line in dump:
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

    def _select_organizations(dump):
        for line in dump:
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
                
            industry = list(format_property(claims, "P452"))
            if industry:
                properties["INDUSTRY"] = industry
            
            ceo = list(format_property(claims, "P169"))
            if ceo:
                properties["CEO"] = ceo
            
            country = list(format_property(claims, "P17"))
            if country:
                properties["COUNTRY"] = country

            legal_form = list(format_property(claims, "P1454"))
            if legal_form:
                properties["LEGAL_FORM"] = legal_form

            if properties["MENTIONS"]:
                yield identifier, properties

    def _select_employed_humans(
        dump: List[dict],
    ) -> Generator[Tuple[str, str], None, None]:
        for line in dump:
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

                first_names = list(format_property(claims, "P734"))
                if first_names:
                    properties["LAST_NAMES"] = first_names

                last_names = list(format_property(claims, "P735"))
                if last_names:
                    properties["FIRST_NAMES"] = last_names

                gender = list(format_property(claims, "P21"))
                if gender:
                    properties["GENDER"] = gender

                occupation = list(format_property(claims, "P106"))
                if occupation:
                    properties["OCCUPATION"] = occupation

                position = list(format_property(claims, "P39"))
                if position:
                    properties["POSITION"] = position

                working_since = list(format_property(claims, "P2031"))
                if working_since:
                    properties["WORKING_SINCE"] = working_since

                owner_of = list(format_property(claims, "P1830"))
                if owner_of:
                    properties["OWNER_OF"] = owner_of

                birth_name = list(format_property(claims, "P1477"))
                if birth_name:
                    properties["MENTIONS"].extend(birth_name)

                if properties["MENTIONS"]:
                    yield identifier, properties

    def _format_properties(
        claims: Dict[str, list], identifier: str
    ) -> Generator[str, None, None]:
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
                        _values.add(value["id"])
                        yield value["text"]
