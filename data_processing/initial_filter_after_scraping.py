import pandas as pd
from datetime import datetime
from data_processing.base_filter import BaseFilter


class InitialAfterScrapingFilter(BaseFilter):
    def fix_auction_columns(self, df):
        mask = df["Auction City Information"] == df["Details"]
        df.loc[mask, "Auction City Information"] = df.loc[mask, "Auction House"]
        df.loc[mask, "Auction House"] = ""
        df.loc[mask, "Auction Date"] = df.loc[mask, "Auction name"]
        df.loc[mask, "Auction name"] = ""
        return df

    def extract_year(self, value):
        value = str(value)
        if "–" in value:
            return value.split("–")[0]
        return value

    def delete_euro_sign(self, value):
        value = str(value)
        if "€" in value:
            euro_index = value.index("€")
            if euro_index + 1 < len(value):
                return value.split("€")[1].strip()
        return value

    def leave_years(self, value):
        value = str(value)
        if "to" in value:
            return value.split("to")[1].strip()
        return value

    def replace_decade_to_year_xx(self, value):
        value = str(value)
        decade = "XX"
        year = "1900"
        if decade in value:
            value = value.replace(decade, year)
        return value

    def replace_question_mark(self, value):
        value = str(value)
        mark = "?"
        if mark in value:
            value = value.replace(mark, "nan")
        return value

    def replace_decade_to_year_xxi(self, value):
        value = str(value)
        decade = "XXI"
        year = "2000"
        if decade in value:
            value = value.replace(decade, year)
        return value

    def replace_decade_to_year_xix(self, value):
        value = str(value)
        decade = "XIX"
        year = "1820"
        if decade in value:
            value = value.replace(decade, year)
        return value

    def delete_unnecessary_sign(self, value):
        for sign in [
            "act.c.",
            "c.",
            "~",
            "(",
            ")",
            "[",
            "]",
            "'",
            "Iki ",
            "po ",
            ", atrib",
            "3.",
            " a. pab.",
        ]:
            value = str(value).replace(sign, "")
        return value

    def keep_after_comma(self, value):
        value = str(value)
        if "," in value:
            value = value.split(",")[-1].strip()
        return value

    def keep_before_comma(self, value):
        value = str(value)
        if "," in value:
            value = value.split(",")[0].strip()
        return value

    def keep_before_space(self, value):
        value = str(value)
        if " " in value:
            value = value.split(" ")[0].strip()
        return value

    def keep_before_slash(self, value):
        value = str(value)
        if "/" in value:
            value = value.split("/")[0].strip()
        return value

    def replace_comma(self, value):
        value = str(value)
        value = value.replace(",", "")
        return value

    def replace_comma_to_dot(self, value):
        value = str(value)
        value = value.replace(",", ".")
        return value

    def replace_nan_to_zero(self, value):
        value = str(value)
        if value == "nan":
            value = 0
        return value

    def replace_not_sold_to_zero(self, value):
        value = str(value)
        if value == "Not sold":
            value = 0
        return value

    def extract_auction_year(self, value):
        value = str(value)
        try:
            splited_value = value.split(" ")
            year = splited_value[-1]
            if not year.isnumeric():
                year = 0
        except:
            year = 0
        return year

    def drop_values_where_is_not_numeric(self, dataset, name):
        dataset[name] = pd.to_numeric(dataset[name], errors="coerce")
        dataset = dataset.dropna(subset=[name])
        dataset[name] = dataset[name].astype(float)

        dataset = dataset.dropna(subset=[name])

    def is_non_numeric(self, value):
        try:
            float(value)
            return False
        except ValueError:
            return value

    def check_where_is_details(self, value):
        value = str(value)
        if "Details" in value:
            return True
        else:
            return False

    def is_numeric(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def replace_non_numeric_to_zero(self, value):
        try:
            float(value)
            return value
        except ValueError:
            return 0

    def replace_non_numeric_to_nan(self, value):
        try:
            float(value)
            return value
        except ValueError:
            return "nan"

    def translate_surface(self, value):
        value = str(value).lower()

        # Dictionary mapping terms to English equivalents, grouped by material type
        translation_map = {
            # Canvas and fabric materials
            "drb. ant kart.": "canvas, cardboard",
            "drb. ant medžio": "canvas, wood",
            "drb. ant fan.": "canvas, plywood",
            "dubl. drb.": "canvas",
            "drobė": "canvas",
            "drb.": "canvas",
            "drb": "canvas",
            "audeklas": "canvas",
            "canvas ant fa.": "canvas, plywood",
            "canvas ant kart": "canvas, cardboard",
            "canvas al.": "canvas",
            "paper ant canvas": "canvas, paper",
            "juodas canvas": "canvas",
            "dub. canvas": "canvas",
            "toile": "canvas",
            "linen": "canvas",
            "belgian linen": "canvas",
            "galacia linen": "canvas",
            "belgian canvas": "canvas",
            "galacia canvas": "canvas",
            "specialos formos drobė": "canvas",
            "drobė ant lentos": "canvas",
            "jute": "canvas",
            "burlap": "canvas",
            "cloth": "canvas",
            "fabric": "canvas",
            "cotton": "canvas",
            "lienzo": "canvas",
            "nioge": "canvas",
            "specialios formos drobė": "canvas",
            # Board materials
            "drb. ant lentos": "canvas, board",
            "lenta": "board",
            "klij.": "board",
            "artist board": "board",
            "artist's board": "board",
            "academy board": "board",
            "clay board": "board",
            "canvasboard": "board",
            "canvas board": "board",
            "woodboard": "board",
            "wood board": "wood, board",
            # Panel materials
            "panel": "panel",
            "panneau": "panel",
            "panneau de chêne": "panel",
            "panneau de chêne parqueté": "panel",
            "oak panel": "panel",
            "wood panel": "panel",
            "panel de chêne": "panel",
            "panel de chêne parqueté": "panel",
            "panel parqueté": "panel",
            # Paper materials
            "popierius": "paper",
            "pop.": "paper",
            "pop": "paper",
            "pop. ant drb.": "paper, canvas",
            "pop. ant kart.": "paper, cardboard",
            "paper guaš.": "paper",
            "papier": "paper",
            "papel": "paper",
            "strong paper": "paper",
            "watercolor paper": "paper",
            "prepared paper": "paper",
            "amatl paper": "paper",
            "amatl handmade bark paper": "paper",
            "ingres": "paper",
            "feuille de cahier": "paper",
            "photograpĥ": "paper",
            "photo": "paper",
            # Cardboard materials
            "kartonas": "cardboard",
            "kart.": "cardboard",
            "cardboard al.": "cardboard",
            "paper ant cardboard": "cardboard",
            "carboard": "cardboard",
            "card": "cardboard",
            "cartón": "cardboard",
            "cartòn": "cardboard",
            "cartone": "cardboard",
            "carton": "cardboard",
            "karton": "cardboard",
            # Wood materials
            "wood": "wood",
            "bois": "wood",
            "amati": "wood",
            "mahogany": "wood",
            "écorce d'eucalyptus": "wood",
            "tabla": "wood",
            "huile": "wood",
            # Plywood/composite materials
            "faniera": "plywood",
            "fan.": "plywood",
            "contreplaqué": "plywood",
            "compensato": "plywood",
            # Hardboards
            "masonite": "hardboard",
            "tablex": "hardboard",
            "platex": "hardboard",
            "isorel": "hardboard",
            "pavatex": "hardboard",
            # Other materials
            "tapetas": "wallpaper",
            "plast.": "plastic",
            "plexiglas": "plastic",
            "skarda": "metal",
            "billy can": "metal",
            "copper": "metal",
            "silk": "fabric",
            "carreau émaillée": "ceramic",
            r'"tabopan"': "chipboard",
            "gauze": "fabric",
        }

        # Case insensitive replacements
        for original, replacement in translation_map.items():
            if original.lower() in value.lower():
                value = value.replace(original.lower(), replacement)

        return value

    def fix_surface(self, value):
        value = str(value)

        if "(" in value:
            value = value.split("(")[0].strip()

        words_to_remove = [
            ". oil",
            " and 4 plastic figurines",
            "thin ",
            ", ³",
            ", ovale",
            ", avec ",
            "thick prepared ",
            "oil",
            "gelatine",
            "gelatina",
            "avec collage",
            "fond serigraphié",
            "with acrylic",
            "»",
            "nepalese ",
            "japan",
            " ant lentos",
            "specialios formos ",
            "treated",
            "ant canvas",
            "collage",
            "unique work",
            " canvasado a mano",
            ")",
            "surface",
        ]

        for word in words_to_remove:
            if word in value.lower():
                value = value.replace(word, "")

        if "ply" in value:
            value = value.replace("ply", "plywood")
        if "plywoodwood" in value:
            value = value.replace("plywoodwood", "plywood")
        if "cardboardboard" in value:
            value = value.replace("cardboardboard", "cardboard")
        if "canvaswood" in value:
            value = value.replace("canvaswood", "canvas, wood")
        if "canvas, canvas" in value:
            value = value.replace("canvas, canvas", "canvas")

        values = [v.strip() for v in value.split(",")]
        values = [v for v in values if v]  # Remove empty strings
        if len(values) > 1:
            value = "|".join(sorted(values))
        else:
            value = values[0] if values else "nan"

        if value == " " or value == ", " or value == "":
            value = "nan"

        return value

    def translate_material(self, value):
        value = str(value).lower()

        translation_map = {
            # Painting techniques
            "temp.": "tempera",
            "temp": "tempera",
            "sint. tempera": "tempera",
            "akr.": "acrylic",
            "akrilas": "acrylic",
            "acrylique": "acrylic",
            "al.": "oil",
            "aliejus": "oil",
            "oil.": "oil,",
            "huile": "oil",
            "akv": "watercolor",
            "akv.": "watercolor",
            "vand. dažai": "watercolor",
            "gouache": "gouache",
            "guaš.": "gouache",
            "guašas": "gouache",
            "enamel": "enamel",
            "lacquer": "varnish",
            "dispersion": "paint",
            "synthetic polymer": "paint",
            "blackboard paint": "paint",
            "alkyd": "paint",
            "painting": "paint",
            "peinture'": "paint",
            "hand painted": "paint",
            "dye and linen": "paint",
            "peinture": "paint",
            "print paint": "paint",
            "pigmentai": "pigment",
            # Drawing techniques
            "anglis": "charcoal",
            "grafitas": "graphite",
            "spalv. piešt.": "pencil",
            "spalv. pencil": "pencil",
            "piešt.": "pencil",
            "kreidelė": "crayon",
            "kreid.": "chalk",
            "craies": "chalk",
            "fusain": "charcoal",
            "pastel et charcoal": "pastel, charcoal",
            "past.": "pastel",
            "past": "pastel",
            "pastelė": "pastel",
            # Printmaking techniques
            "ofort.": "etching",
            "šilkograf.": "silkscreen",
            "moliotip.": "lithograph",
            "monotip.": "monotype",
            "montip.": "monotype",
            "monotipija": "monotype",
            "raiž.": "engraving",
            "atsp.": "print",
            "printing color": "print",
            "poligraf.": "print",
            "lithograph": "lithograph",
            "etching": "etching",
            "graviūra": "engraving",
            # Mixed media and collage
            "aut. techn.": "mixed media",
            "mišri techn.": "mixed media",
            "mišri tech.": "mixed media",
            "mišri autorinė techn.": "mixed media",
            "technique mixte": "mixed media",
            "koliažas": "collage",
            "koliaž.": "collage",
            "collage": "collage",
            "collages": "collage",
            "collage et acrylic": "collage, acrylic",
            "laikraščio iškarpos": "collage",
            "collage et acrylique": "collage, acrylic",
            # Other techniques
            "tušas": "ink",
            "tuš.": "ink",
            "ink": "ink",
            "smėlis": "sand",
            "sand": "sand",
            "pigment": "pigment",
            "bronze foil": "foil",
            "bronzos folija": "foil",
            "gold leaf": "foil",
            "heightened with foil": "foil",
            "feuille d’or": "foil",
            "folija": "foil",
            "heightened with metal": "metal",
            "aerozoliniai dažai": "spray paint",
            "gold spray": "spray paint",
            "ate à modeler": "clay",
            "minkštas lakas": "varnish",
            "lakas": "varnish",
        }

        for original, replacement in translation_map.items():
            if original in value:
                value = value.replace(original, replacement)

        return value

    def fix_material(self, value):
        value = str(value)

        if "(" in value:
            value = value.split("(")[0].strip()

        words_to_remove = [
            ". unique work",
            "application",
            "30",
            ".",
            "kartono ",
            "tekstolito ",
            "ant kart.",
            "ant kart",
            "kart.",
        ]

        for word in words_to_remove:
            if word in value.lower():
                value = value.replace(word, "")

        if "pastelel" in value:
            value = value.replace("pastelel", "pastel")
        if "pclay" in value:
            value = value.replace("pclay", "clay")
        if "oil pastel" in value:
            value = value.replace("oil pastel", "pastel, oil")
        if "oil chalk" in value:
            value = value.replace("oil chalk", "chalk, oil")
        if "chalk pastel" in value:
            value = value.replace("chalk pastel", "pastel, chalk")
        if "temperaera" in value:
            value = value.replace("temperaera", "tempera")
        if "pastelė" in value:
            value = value.replace("pastelė", "pastel")
        if value == "al":
            value = "oil"

        values = [v.strip() for v in value.split(",")]
        values = [v for v in values if v]  # Remove empty strings
        if len(values) > 1:
            value = "|".join(sorted(values))
        else:
            value = values[0] if values else "nan"

        if value == " " or value == ", " or value == "":
            value = "nan"

        return value

    def lowercase(self, value):
        value = str(value)
        value = value.lower()
        return value

    def split_techniques(self, dataset):
        dataset["Materials"] = ""
        dataset["Surface"] = ""
        # print(len(dataset))
        for i in range(len(dataset)):
            element = dataset.iloc[i]
            description = element.iloc[5]
            if element.iloc[13] == "Vilniaus Aukcionas":
                if isinstance(description, str) and description:
                    if "," in description:
                        array = description.split(", ")
                    else:
                        array = [description]
                else:
                    array = []
                array_length = len(array)
                if array:
                    dataset.at[i, "Surface"] = array[0]
                    if len(array) > 1:
                        dataset.at[i, "Materials"] = ", ".join(array[1:])
                    else:
                        dataset.at[i, "Materials"] = "nan"
                else:
                    dataset.at[i, "Surface"] = "nan"
                    dataset.at[i, "Materials"] = "nan"

            else:
                if isinstance(description, str) and description:
                    if "/" in description:
                        array = description.split("/")
                    else:
                        array = [description]
                else:
                    array = []
                array_length = len(array)

                if array:
                    dataset.at[i, "Materials"] = array[0]
                    if len(array) > 1:
                        dataset.at[i, "Surface"] = "/".join(array[1:])
                    else:
                        dataset.at[i, "Surface"] = "nan"
                else:
                    dataset.at[i, "Materials"] = "nan"
                    dataset.at[i, "Surface"] = "nan"
        return dataset

    def replace_slash_to_comma(self, value):
        value = str(value)
        value = value.replace("/", ", ")
        return value

    def leave_just_country(self, value):
        value = str(value)
        array = value.split(", ")
        length = len(array)
        result = array[length - 1]
        return result

    def get_time_pasted_from_auction(self, value):
        current_year = datetime.now().year
        time = int(current_year) - int(value)
        return time

    def get_time_pasted_from_creation_till_auction(self, auction, creation):
        if auction == "nan" or creation == "nan":
            time = "nan"
        else:
            # print(f"{auction} = {creation}")

            time = int(auction) - int(creation)
        return time

    def is_dead(self, birth, death):
        if death == "nan":
            if birth != "nan":
                current_year = datetime.now().year
                time = int(current_year) - int(birth)
                if int(time) > 100:
                    is_death_value = True
                else:
                    is_death_value = False
            else:
                is_death_value = "nan"
        else:
            is_death_value = True
        return is_death_value

    def artist_lifetime_year(self, birth, death):
        if death == "nan":
            if birth != "nan":
                current_year = datetime.now().year
                time = int(current_year) - int(birth)
                if int(time) > 100:
                    years = 100
                else:
                    years = 0
            else:
                years = 0
        else:
            if birth != "nan":
                years = int(death) - int(birth)
            else:
                years = 0
        return years

    def artist_years_now(self, birth, death):
        current_year = datetime.now().year
        if death == "nan":
            if birth != "nan":
                years = int(current_year) - int(birth)
            else:
                years = 0
        else:
            years = 0
        return years

    def get_authors_age(self, birth, death):
        if birth == "nan" or death == "nan":
            time = "nan"
        else:
            time = int(death) - int(birth)
        return time

    def get_area(self, width, height):
        area = 0
        if width != "nan" or height != "nan":
            area = float(width) * float(height)
        return round(area, 2)

    def calculate_min_max_average(self, full_list):
        full_list["Sold Price"] = pd.to_numeric(
            full_list["Sold Price"], errors="coerce"
        )
        full_list = full_list.dropna(subset=["Sold Price"])
        unique_set = set(full_list["Artist name"])
        unique_count = len(unique_set)

        artist_data = {}

        for i in range(len(full_list)):
            one = full_list.iloc[i]
            artist = one["Artist name"]
            sold_price = one["Sold Price"]

            if artist not in artist_data:
                artist_data[artist] = {
                    "total_price": 0,
                    "count": 0,
                    "min_price": float("inf"),
                    "max_price": float("-inf"),
                }

            artist_data[artist]["total_price"] += sold_price
            artist_data[artist]["count"] += 1
            artist_data[artist]["min_price"] = min(
                artist_data[artist]["min_price"], sold_price
            )
            artist_data[artist]["max_price"] = max(
                artist_data[artist]["max_price"], sold_price
            )

        full_list["Average Sold Price"] = 0.0
        full_list["Min Sold Price"] = 0.0
        full_list["Max Sold Price"] = 0.0

        for artist, data in artist_data.items():
            average_price = round(data["total_price"] / data["count"], 2)

            full_list.loc[full_list["Artist name"] == artist, "Average Sold Price"] = (
                average_price
            )
            full_list.loc[full_list["Artist name"] == artist, "Min Sold Price"] = data[
                "min_price"
            ]
            full_list.loc[full_list["Artist name"] == artist, "Max Sold Price"] = data[
                "max_price"
            ]

        print(
            full_list[
                [
                    "Artist name",
                    "Average Sold Price",
                    "Min Sold Price",
                    "Max Sold Price",
                ]
            ]
        )
        return full_list

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:

        df = self.fix_auction_columns(df)

        df["Description"] = df["Description"].apply(self.lowercase)
        df = self.split_techniques(df)
        df["Surface"] = df["Surface"].apply(self.translate_surface)
        df["Surface"] = df["Surface"].apply(self.replace_slash_to_comma)
        df["Surface"] = df["Surface"].apply(self.fix_surface)
        df["Materials"] = df["Materials"].apply(self.translate_material)
        df["Materials"] = df["Materials"].apply(self.replace_slash_to_comma)
        df["Materials"] = df["Materials"].apply(self.fix_material)

        words_to_find = [
            "sign.",
            "signed",
            "Signed",
            "Sign.",
            "AK",
            "AD",
            "VK",
            "VD",
            "KP",
            "DP",
        ]
        df["Signed"] = df["Details"].apply(
            lambda x: any(word in x for word in words_to_find) if pd.notna(x) else False
        )
        df.drop("Auction name", axis=1, inplace=True)
        df["Creation Year"] = df["Creation Year"].apply(self.extract_year)
        df["Estimated Minimum Price"] = df["Estimated Minimum Price"].apply(
            self.delete_euro_sign
        )
        df["Sold Price"] = df["Sold Price"].apply(self.delete_euro_sign)
        df["Estimated Maximum Price"] = df["Estimated Maximum Price"].apply(
            self.delete_euro_sign
        )
        df["Estimated Minimum Price"] = df["Estimated Minimum Price"].apply(
            self.replace_nan_to_zero
        )
        df["Estimated Maximum Price"] = df["Estimated Maximum Price"].apply(
            self.replace_nan_to_zero
        )
        df["Sold Price"] = df["Sold Price"].apply(self.replace_not_sold_to_zero)
        df["Auction Date"] = df["Auction Date"].apply(self.extract_auction_year)
        df["Sold Price"] = df["Sold Price"].apply(self.replace_comma)
        df["Estimated Maximum Price"] = df["Estimated Maximum Price"].apply(
            self.replace_comma
        )
        df["Estimated Minimum Price"] = df["Estimated Minimum Price"].apply(
            self.replace_comma
        )
        df["Creation Year"] = df["Creation Year"].apply(self.delete_unnecessary_sign)
        df["Creation Year"] = df["Creation Year"].apply(self.keep_after_comma)
        df["Artist Birth Year"] = df["Artist Birth Year"].apply(
            self.delete_unnecessary_sign
        )
        df["Artist Birth Year"] = df["Artist Birth Year"].apply(
            self.replace_non_numeric_to_nan
        )
        df["Artist Birth Year"] = df["Artist Birth Year"].apply(self.keep_after_comma)
        df["Artist Birth Year"] = df["Artist Birth Year"].apply(self.keep_before_slash)
        df["Artist Death Year"] = df["Artist Death Year"].apply(
            self.delete_unnecessary_sign
        )
        df["Artist Death Year"] = df["Artist Death Year"].apply(self.keep_before_slash)
        df["Artist Death Year"] = df["Artist Death Year"].apply(self.keep_before_comma)
        df["Creation Year"] = df["Creation Year"].apply(self.replace_decade_to_year_xx)
        df["Creation Year"] = df["Creation Year"].apply(self.replace_non_numeric_to_nan)
        df["Artist Birth Year"] = df["Artist Birth Year"].apply(
            self.replace_decade_to_year_xxi
        )
        df["Artist Death Year"] = df["Artist Death Year"].apply(
            self.replace_decade_to_year_xxi
        )
        df["Artist Birth Year"] = df["Artist Birth Year"].apply(
            self.replace_decade_to_year_xx
        )
        df["Artist Death Year"] = df["Artist Death Year"].apply(
            self.replace_decade_to_year_xx
        )
        df["Artist Birth Year"] = df["Artist Birth Year"].apply(
            self.replace_decade_to_year_xix
        )
        df["Artist Birth Year"] = df["Artist Birth Year"].apply(
            self.replace_question_mark
        )
        df["Artist Death Year"] = df["Artist Death Year"].apply(
            self.replace_question_mark
        )
        df["Width"] = df["Width"].apply(self.replace_comma_to_dot)
        df["Width"] = df["Width"].apply(self.keep_before_space)
        df["Width"] = df["Width"].apply(self.replace_non_numeric_to_nan)
        df["Height"] = df["Height"].apply(self.replace_comma_to_dot)
        df["Height"] = df["Height"].apply(self.keep_before_space)
        df["Height"] = df["Height"].apply(self.replace_non_numeric_to_nan)
        df = df.drop(columns=["Details"])
        df = df[df["Sold Price"].apply(self.is_numeric)]
        df = df[df["Estimated Minimum Price"].apply(self.is_numeric)]
        df["Estimated Maximum Price"] = df["Estimated Maximum Price"].apply(
            self.replace_non_numeric_to_zero
        )
        df = df.drop(columns=["Description"])
        df["Auction Country"] = df["Auction City Information"].apply(
            self.leave_just_country
        )
        df["Auction Date"] = df["Auction Date"].apply(self.leave_years)
        df["Years from auction till now"] = df["Auction Date"].apply(
            self.get_time_pasted_from_auction
        )
        df["Years from creation till auction"] = df.apply(
            lambda row: self.get_time_pasted_from_creation_till_auction(
                row["Auction Date"], row["Creation Year"]
            ),
            axis=1,
        )
        df["Years from birth till creation"] = df.apply(
            lambda row: self.get_time_pasted_from_creation_till_auction(
                row["Creation Year"], row["Artist Birth Year"]
            ),
            axis=1,
        )
        df["Is dead"] = df.apply(
            lambda row: self.is_dead(
                row["Artist Birth Year"], row["Artist Death Year"]
            ),
            axis=1,
        )
        df["Artist Lifetime"] = df.apply(
            lambda row: self.artist_lifetime_year(
                row["Artist Birth Year"], row["Artist Death Year"]
            ),
            axis=1,
        )
        df["Artist Years Now"] = df.apply(
            lambda row: self.artist_years_now(
                row["Artist Birth Year"], row["Artist Death Year"]
            ),
            axis=1,
        )
        df["Area"] = df.apply(
            lambda row: self.get_area(row["Width"], row["Height"]), axis=1
        )
        # df = df[df["Sold Price"].astype(float) > 0]

        # df = df.drop(columns=['Artist Death Year'])
        # df = df.drop(columns=['Artist Birth Year'])
        # df = df.drop(columns=['Creation Year'])
        df = df.drop(columns=["Primary Price"])
        df = df.drop(columns=["Auction Date"])
        df = df.drop(columns=["Auction City Information"])

        # df = remove_outliers_by_sold_price(df, column="Sold Price", method="iqr", factor=2)
        df = self.calculate_min_max_average(df)

        return df
