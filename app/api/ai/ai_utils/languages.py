from enum import Enum
import numpy as np


class Language(Enum):
    Arabic = 0
    Basque =1
    Breton =2
    Catalan =3
    Chinese_China =4
    Chinese_Hongkong =5
    Chinese_Taiwan =6
    Chuvash =7
    Czech =8
    Dhivehi =9
    Dutch =10
    English =11
    Esperanto =12
    Estonian =13
    French =14
    Frisian =15
    Georgian =16
    German =17
    Greek =18
    Hakha_Chin =19
    Indonesian =20
    Interlingua =21
    Italian =22
    Japanese =23
    Kabyle =24
    Kinyarwanda =25
    Kyrgyz =26
    Latvian =27
    Maltese =28
    Mongolian =29
    Persian =30
    Polish =31
    Portuguese =32
    Romanian =33
    Romansh_Sursilvan=34
    Russian = 35
    Sakha =36
    Slovenian =37
    Spanish =38
    Swedish =39
    Tamil =40
    Tatar =41
    Turkish =42
    Ukrainian =43
    Welsh = 44


language_codes = np.array(
        [("Arabic", "ar-sa", "ar"),
         ("English", 'en-AU', 'en-US'),
         ("English", 'en-GB', 'en'),
         ("Swedish", 'sv-SE', 'sv'),
         ("Catalan", "ca-ES", "ca"),
         ("Chinese", "zh-CN", "zh"),
         ("Chinese", "zh-Hant", "zh"),
         ("Czech", "cs-CZ", "cz"),
         ("Dutch", "nl-BE", "nl"), #nl added not in list
         ("Estonian", "et-EE", "et"),
         ("French", "fr-FR", "fr"),
         ("German", "de-AT", "de"),
         ("Greek", "el-GR", "el"),
         ("Indonesian", "id", "id"),
         ("Italian", "it-IT", "it"),
         ("Japanese", "ja-JP", "ja"),
         ("Latvian", "lv-LV", "lv"),
         ("Polish", "pl-PL", "pl"),
         ("Portuguese", "pt-PT", "pt"),
         ("Romanian", "ro-RO", "ro"),
         ("Russian", "ru-RU", "ru"),
         ("Ukrainian", "uk", "uk"), #Ukrainian
         ("Spanish", "es-ES", "es"),
         ("Turkish", "tr-TR", "tr"),
         ("Welsh", "unknown", "cy")],
        dtype=[('id', 'U10'), ('code1', 'U10'), ('code2', 'U10')])


def get_language_code(index):
    result_language = Language(index).name

    # check for matching rows in languages_ISO
    matching_row = language_codes[language_codes['id'] == result_language]

    # If the languages is not supported, see languages_ISO, return not supported
    if len(matching_row) == 0:
        return ["Not supported", "If supported, try a longer sentence.", "0"]

    language_code = matching_row['code2'][0].item()

    return [result_language, language_code]
