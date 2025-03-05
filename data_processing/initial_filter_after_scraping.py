import pandas as pd
from datetime import datetime
import sys
from data_processing.base_filter import BaseFilter
sys.stdout.reconfigure(encoding='utf-8')

class InitialAfterScrapingFilter(BaseFilter):
    def extract_year(value):
        value = str(value)
        if '–' in value:
            return value.split('–')[0]
    return value  

    def delete_euro_sign(value):
        value = str(value)
        if '€' in value:
            euro_index = value.index('€')
            if euro_index + 1 < len(value):
                return value.split('€')[1].strip()
        return value 

    def leave_years(value):
        value = str(value)
        if 'to' in value:
            return value.split('to')[1].strip()
        return value 

    def replace_decade_to_year_xx(value):
        value = str(value)
        decade = "XX"
        year = "1900"
        if decade in value:
            value = value.replace(decade,year)
        return value

    def replace_question_mark(value):
        value = str(value)
        mark = "?"
        if mark in value:
            value = value.replace(mark,"nan")
        return value

    def replace_decade_to_year_xxi(value):
        value = str(value)
        decade = "XXI"
        year = "2000"
        if decade in value:
            value = value.replace(decade,year)
        return value

    def replace_decade_to_year_xix(value):
        value = str(value)
        decade = "XIX"
        year = "1820"
        if decade in value:
            value = value.replace(decade,year)
        return value

    def delete_unnecessary_sign(value):
        value = str(value)
        if 'act.c.' in value:
            value = value.replace("act.c.","") 
        if 'c.' in value:
            value = value.replace("c.","") 
        if '~' in value:
            value = value.replace("~","")     
        if '(' in value:
            value = value.replace("(","")  
        if ')' in value:
            value = value.replace(")","")  
        if '[' in value:
            value = value.replace("[","")  
        if ']' in value:
            value = value.replace("]","")         
        if "'" in value:
            value = value.replace("'","")    
        if "Iki " in value:
            value = value.replace("Iki ","")
                
        return value 
    
    def replace_comma(value):
        value = str(value)
        value = value.replace(",","")
        return value 
    
    def replace_comma_to_dot(value):
        value = str(value)
        value = value.replace(",",".")
        return value 

    def replace_nan_to_zero(value):
        value = str(value)
        if value == 'nan':
            value=0
        return value 

    def replace_not_sold_to_zero(value):
        value = str(value)
        if value == 'Not sold':
            value=0
        return value 

    def extract_auction_year(value):
        value = str(value)
        try:
            splited_value = value.split(' ')
            year = splited_value[-1] 
            if not year.isnumeric():
                year = 0 
        except:
            year =  0 
        return year 

    def drop_values_where_is_not_numeric(dataset, name):
        dataset[name] = pd.to_numeric(dataset[name], errors='coerce')
        dataset = dataset.dropna(subset=[name])
        dataset[name] = dataset[name].astype(float)

        dataset = dataset.dropna(subset=[name])
        
    def is_non_numeric(value):
        try:
            float(value)  
            return False
        except ValueError:
            return value

    def check_where_is_details(value):
        value = str(value)
        if "Details" in value:
            return True
        else:
            return False
    def is_numeric(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def replace_non_numeric_to_zero(value):
        try:
            float(value)  
            return value
        except ValueError:        
            return 0
        
    def translate_surface(value):
        value = str(value)
        if 'drb. ant kart.' in value:
            value = value.replace("drb. ant kart.","canvas")  
        if 'drb. ant lentos' in value:
            value = value.replace("drb. ant lentos","board") 
        if 'specialios formos drobė' in value:
            value = value.replace("specialios formos drobė","canvas")  
        if 'drb. ant medžio' in value:
            value = value.replace("drb. ant medžio","canvas")    
        if 'drb. ant fan.' in value:
            value = value.replace("drb. ant fan.","canvas")       
        if 'pop. (lotyniško bažnytinio giesmyno puslapis)' in value:
            value = value.replace("pop. (lotyniško bažnytinio giesmyno puslapis)","paper")     
        if 'dubl. drb.' in value:
            value = value.replace("dubl. drb.","canvas")        
        if 'popierius' in value:
            value = value.replace("popierius","paper") 
        if 'Pop.' in value:
            value = value.replace("Pop.","paper") 
        if 'pop.' in value:
            value = value.replace("pop.","paper") 
        if 'pop' in value:
            value = value.replace("pop","paper") 
        if 'drobė' in value:
            value = value.replace("drobė","canvas")   
        if 'drb.' in value:
            value = value.replace("drb.","canvas")   
        if 'Drb.' in value:
            value = value.replace("Drb.","canvas")  
        if 'audeklas' in value:
            value = value.replace("audeklas","canvas")   
        if 'drb' in value:
            value = value.replace("drb.","canvas")  
        if 'faniera' in value:
            value = value.replace("faniera","plywood") 
        if 'fan.' in value:
            value = value.replace("fan.","plywood")  
        if 'Fan.' in value:
            value = value.replace("Fan.","plywood") 
        if 'lenta' in value:
            value = value.replace("lenta","board")  
        if 'klij.' in value:
            value = value.replace("klij.","board")  
        if 'kartonas' in value:
            value = value.replace("kartonas","cardboard") 
        if 'kart.' in value:
            value = value.replace("kart.","cardboard") 
        if 'Kart.' in value:
            value = value.replace("Kart.","cardboard") 
        if 'tapetas' in value:
            value = value.replace("tapetas","wallpaper") 
        if 'plast.' in value:
            value = value.replace("plast.","plastic") 
        if 'skarda' in value:
            value = value.replace("skarda","metal") 
        if 'pop. ant drb.' in value:
            value = value.replace("pop. ant drb.","paper")     
        if 'pop. ant kart.' in value:
            value = value.replace("pop. ant kart.","paper") 
        return value

    def translate_material(value):
        value = str(value)
        if 'temp.' in value:
            value = value.replace("temp.","tempera")
        if 'guaš.' in value:
            value = value.replace("guaš.","gouache") 
        if 'guašas' in value:
            value = value.replace("guašas","gouache") 
        if 'mišri techn.' in value:
            value = value.replace("mišri techn.","mixed media") 
        if 'mišri tech.' in value:
            value = value.replace("mišri tech.","mixed media") 
        if 'anglis' in value:
            value = value.replace("anglis","charcoal")  
        if 'tušas' in value:
            value = value.replace("tušas","ink")  
        if 'tuš.' in value:
            value = value.replace("tuš.","ink")  
        if 'akr.' in value:
            value = value.replace("akr.","acrylic")  
        if 'akrilas' in value:
            value = value.replace("akrilas","acrylic")  
        if 'past.' in value:
            value = value.replace("past.","pastel")  
        if 'pastelė' in value:
            value = value.replace("pastelė","pastel")  
        if 'koliažas' in value:
            value = value.replace("koliažas","collage")   
        if 'koliaž.' in value:
            value = value.replace("koliaž.","collage") 
        if 'akv.' in value:
            value = value.replace("akv.","watercolor")          
        if 'grafitas' in value:
            value = value.replace("grafitas","graphite")   
        if 'ofort.' in value: 
            value = value.replace("ofort.","etching") 
        if 'dažai' in value: 
            value = value.replace("dažai","spray paint") 
        if 'šilkograf.' in value: 
            value = value.replace("šilkograf.","silkscreen")
        if 'piešt.' in value: 
            value = value.replace("piešt.","pencil")
        if 'lakas' in value:
            value = value.replace("lakas","varnish") 
        if 'moliotip.' in value:
            value = value.replace("moliotip.","lithograph")     
        if 'smėlis' in value:
            value = value.replace("smėlis","sand")   
        if 'raiž.' in value:
            value = value.replace("raiž.","engraving")        
        if 'atsp.' in value:
            value = value.replace("atsp.","print")    
        if 'al.' in value:
            value = value.replace("al.","oil") 
        if 'aliejus'  in value:
            value = value.replace("aliejus","oil") 
        if 'monotip.'  in value:
            value = value.replace("monotip.","monotype")
        if 'kreidelė' in value:
            value = value.replace("kreidelė","chalk") 
        if 'kreid.' in value:
            value = value.replace("kreid.","chalk")   
        return value

    def lowercase(value):
        value = str(value)
        value = value.lower()
        return value 
    
    def split_techniques(dataset):
        dataset['Materials'] = ""
        dataset['Surface'] =""
        print(len(dataset))
        for i in range(len(dataset)):
            element = dataset.iloc[i]
            description = element.iloc[5]
            if element.iloc[13] == 'Vilniaus Aukcionas':
                if isinstance(description, str) and description:
                    if ',' in description:
                        array = description.split(', ')                
                    else:
                        array = [description]
                else:
                    array=[]
                array_length = len(array)
                if array:
                    dataset.at[i, 'Surface'] = array[0] 
                    if len(array) > 1:
                        dataset.at[i, 'Materials'] = ', '.join(array[1:])  
                    else:
                        dataset.at[i, 'Materials'] = "nan" 
                else:
                    dataset.at[i, 'Surface'] = "nan" 
                    dataset.at[i, 'Materials'] = "nan"  
                    
            else:
                if isinstance(description, str) and description:
                    if '/' in description:
                        array = description.split('/')                
                    else:
                        array = [description]
                else:
                    array=[]
                array_length = len(array)
                
                if array:
                    dataset.at[i, 'Materials'] = array[0] 
                    if len(array) > 1:
                        dataset.at[i, 'Surface'] = '/'.join(array[1:])  
                    else:
                        dataset.at[i, 'Surface'] = "nan" 
                else:
                    dataset.at[i, 'Materials'] = "nan" 
                    dataset.at[i, 'Surface'] = "nan"  
        return dataset

    def replace_slash_to_comma(value):
        value = str(value)
        value = value.replace("/",", ")
        return value 


    def leave_just_country(value):
        value = str(value)
        array = value.split(', ')  
        length = len(array)
        result = array[length-1]
        return result  
        
    def get_time_pasted_from_auction(value):
        current_year = datetime.now().year 
        time = int(current_year) - int(value)
        return time   

    def get_time_pasted_from_creation_till_auction(auction, creation):
        if auction == 'nan' or creation ==  'nan':
            time='nan'
        else:     
            print(f"{auction} = {creation}")

            time =  int(auction) - int(creation)
        return time   

    def is_dead(birth, death):
        if death == 'nan':
            if birth != 'nan':
                current_year = datetime.now().year
                time = int(current_year) - int(birth)
                if int(time) > 100:
                    is_death_value = True
                else:
                    is_death_value = False
            else:
                is_death_value = 'nan'
        else: 
            is_death_value = True
        return is_death_value   

    def artist_lifetime_year(birth, death):
        if death == 'nan':
            if birth != 'nan':
                current_year = datetime.now().year
                time = int(current_year) - int(birth)
                if int(time) > 100:
                    years = 100
                else:
                    years = 0
            else:
                years = 0
        else: 
            if birth != 'nan':
                years = int(death) - int(birth)
            else:
                years = 0
        return years   

    def artist_years_now(birth, death):
        current_year = datetime.now().year
        if death == 'nan':
            if birth != 'nan':            
                years = int(current_year) - int(birth)
            else:
                years = 0
        else: 
            years = 0
        return years 
    
    def get_authors_age(birth, death):
        if auction == 'nan' or creation ==  'nan':
            time='nan'
        else:     
            time = int(death) - int(birth)
        return time  
     
    def get_area(width, height):
        area = 0
        if width!='nan' or height != "nan":
            area = float(width)*float(height) 
        return round(area,2)       

    def calculate_min_max_average(full_list):
        full_list['Sold Price'] = pd.to_numeric(full_list['Sold Price'], errors='coerce') 
        full_list = full_list.dropna(subset=['Sold Price']) 
        unique_set = set(full_list['Artist name'])
        unique_count = len(unique_set)

        artist_data = {}  
        
        for i in range(len(full_list)):
            one = full_list.iloc[i]
            artist = one['Artist name']
            sold_price = one['Sold Price']

            if artist not in artist_data:
                artist_data[artist] = {'total_price': 0, 'count': 0, 'min_price': float('inf'), 'max_price': float('-inf')}
                
            artist_data[artist]['total_price'] += sold_price
            artist_data[artist]['count'] += 1
            artist_data[artist]['min_price'] = min(artist_data[artist]['min_price'], sold_price)
            artist_data[artist]['max_price'] = max(artist_data[artist]['max_price'], sold_price)

        full_list["Average Sold Price"] = 0
        full_list["Min Sold Price"] = 0
        full_list["Max Sold Price"] = 0

        for artist, data in artist_data.items():
            average_price = round(data['total_price'] / data['count'], 2)

            full_list.loc[full_list['Artist name'] == artist, 'Average Sold Price'] = average_price
            full_list.loc[full_list['Artist name'] == artist, 'Min Sold Price'] = data['min_price']
            full_list.loc[full_list['Artist name'] == artist, 'Max Sold Price'] = data['max_price']

        print(full_list[['Artist name', 'Average Sold Price', 'Min Sold Price', 'Max Sold Price']])
        return full_list
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Description'] = df['Description'].apply(lowercase)
        df = split_techniques(df)
        df['Surface'] = df['Surface'].apply(translate_surface)
        df['Materials'] = df['Materials'].apply(translate_material)
        words_to_find = ['sign.', 'signed', 'Signed', 'Sign.']
        df['Signed'] = df['Details'].apply(lambda x: any(word in x for word in words_to_find) if pd.notna(x) else False)
        df.drop('Auction name', axis=1, inplace=True)
        df['Creation Year'] = df['Creation Year'].apply(extract_year)
        df['Estimated Minimum Price'] = df['Estimated Minimum Price'].apply(delete_euro_sign)
        df['Sold Price'] = df['Sold Price'].apply(delete_euro_sign)
        df['Estimated Maximum Price'] = df['Estimated Maximum Price'].apply(delete_euro_sign)
        df['Estimated Minimum Price'] = df['Estimated Minimum Price'].apply(replace_nan_to_zero)
        df['Estimated Maximum Price'] = df['Estimated Maximum Price'].apply(replace_nan_to_zero)
        df['Sold Price'] = df['Sold Price'].apply(replace_not_sold_to_zero)
        df['Auction Date'] = df['Auction Date'].apply(extract_auction_year)
        df['Sold Price'] = df['Sold Price'].apply(replace_comma)
        df['Estimated Maximum Price'] = df['Estimated Maximum Price'].apply(replace_comma)
        df['Estimated Minimum Price'] = df['Estimated Minimum Price'].apply(replace_comma)
        df['Creation Year'] = df['Creation Year'].apply(delete_unnecessary_sign)
        df['Artist Birth Year'] = df['Artist Birth Year'].apply(delete_unnecessary_sign)
        df['Artist Death Year'] = df['Artist Death Year'].apply(delete_unnecessary_sign)
        df['Creation Year'] = df['Creation Year'].apply(replace_decade_to_year_xx)
        df['Artist Birth Year'] = df['Artist Birth Year'].apply(replace_decade_to_year_xxi)
        df['Artist Death Year'] = df['Artist Death Year'].apply(replace_decade_to_year_xxi)
        df['Artist Birth Year'] = df['Artist Birth Year'].apply(replace_decade_to_year_xx)
        df['Artist Death Year'] = df['Artist Death Year'].apply(replace_decade_to_year_xx)
        df['Artist Birth Year'] = df['Artist Birth Year'].apply(replace_decade_to_year_xix)
        df['Artist Birth Year'] = df['Artist Birth Year'].apply(replace_question_mark)
        df['Artist Death Year'] = df['Artist Death Year'].apply(replace_question_mark)
        df['Width'] = df['Width'].apply(replace_comma_to_dot)
        df['Height'] = df['Height'].apply(replace_comma_to_dot)
        df = df.drop(columns=['Details'])
        df = df[df['Sold Price'].apply(is_numeric)]
        df = df[df['Estimated Minimum Price'].apply(is_numeric)]
        df['Estimated Maximum Price'] = df['Estimated Maximum Price'].apply(replace_non_numeric_to_zero)
        df['Surface'] = df['Surface'].apply(replace_slash_to_comma)
        df = df.drop(columns=['Description'])
        df['Auction Country'] = df['Auction City Information'].apply(leave_just_country)        
        df['Auction Date'] = df['Auction Date'].apply(leave_years)        
        df['Years from auction till now'] = df['Auction Date'].apply(get_time_pasted_from_auction)        
        df['Years from creation till auction'] = df.apply(lambda row: get_time_pasted_from_creation_till_auction(row['Auction Date'], row['Creation Year']), axis=1)
        df['Years from birth till creation'] = df.apply(lambda row: get_time_pasted_from_creation_till_auction(row['Creation Year'], row['Artist Birth Year']), axis=1)
        df['Is dead'] = df.apply(lambda row: is_dead(row['Artist Birth Year'], row['Artist Death Year']), axis=1)
        df['Artist Lifetime'] = df.apply(lambda row: artist_lifetime_year(row['Artist Birth Year'], row['Artist Death Year']), axis=1)
        df['Artist Years Now'] = df.apply(lambda row: artist_years_now(row['Artist Birth Year'], row['Artist Death Year']), axis=1)
        df['Area'] = df.apply(lambda row: get_area(row['Width'], row['Height']),axis=1)
        df = df[df['Sold Price'].astype(float) > 0]
        
        # df = df.drop(columns=['Artist Death Year'])
        # df = df.drop(columns=['Artist Birth Year'])
        # df = df.drop(columns=['Creation Year'])
        df = df.drop(columns=['Auction Date'])
        df = df.drop(columns=['Auction City Information'])        
              
        # df = remove_outliers_by_sold_price(df, column="Sold Price", method="iqr", factor=2)        
        df = calculate_min_max_average(df)
        
        return df