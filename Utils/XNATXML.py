import xml.etree.cElementTree as ET
import requests
import pandas as pd
from io import StringIO

### Class to automatic create XML search from specified entry criteria and conditions using pycURL to be used for XNAT
### To explore available data, use curl -u user:pass -X GET  http://128.16.11.124:8080/xnat/data/search/elements/?format=csv

class XMLCreator():
    def __init__(self, root_element, search_field_list, search_where_list):

        self.bundle = ET.Element("xdat:bundle",
                                 {"ID":"@xnat:subjectData",
                                  "xmlns:arc":"http://nrg.wustl.edu/arc",
                                  "xmlns:cat":"http://nrg.wustl.edu/catalog",
                                  "xmlns:pipe":"http://nrg.wustl.edu/pipe",
                                  "xmlns:prov":"http:/ www.nbirn.net/prov",
                                  "xmlns:scr":"http://nrg.wustl.edu/scr",
                                  "xmlns:val":"http://nrg.wustl.edu/val",
                                  "xmlns:wrk":"http://nrg.wustl.edu/workflow",
                                  "xmlns:xdat":"http://nrg.wustl.edu/security",
                                  "xmlns:xnat":"http://nrg.wustl.edu/xnat",
                                  "xmlns:xnat_a":"http://nrg.wustl.edu/xnat_assessments",
                                  "xmlns:xsi":"http://www.w3.org/2001/XMLSchema-instance"})        
        
        self.search_field_list = search_field_list
        self.root_element      = root_element
        self.search_where_list = search_where_list
    def ConstructTree(self):
        
        root_element = ET.SubElement(self.bundle,"xdat:root_element_name").text = self.root_element ## Search uniquely against those 
        
        for search_field in self.search_field_list:
            self.search_field_query(self.bundle, search_field)
                
        search_where_clause = ET.SubElement(self.bundle,"xdat:search_where", {"method":"AND"})                    
        for search_where in self.search_where_list:
            self.search_where_query(search_where_clause, search_where)            
                
        tree = ET.ElementTree(self.bundle)                    
        ET.indent(tree, '  ')
                    
        #writing xml
        tree.write("example.xml", encoding="utf-8", xml_declaration=True)

    def search_field_query(self,bundle,search_field):
        root = ET.SubElement(bundle,"xdat:search_field")
        for key,value in search_field.items():
            ET.SubElement(root, "xdat:"+key).text = value

    def search_where_query(self, search_where_clause, search_where):
        criteria = ET.SubElement(search_where_clause, "xdat:criteria")
        for key,value in search_where.items():
            ET.SubElement(criteria, "xdat:"+key).text = value
    
if __name__ == "__main__":
    search_field = [{"element_name":"xnat:subjectData","field_ID":"SUBJECT_LABEL","sequence":"1", "type":"1"}]
    search_where = [{"schema_field":"xnat:subjectData.XNAT_SUBJECTDATA_FIELD_MAP=survival_status","comparison_type":"=","value":"1"},
                    {"schema_field":"xnat:ctSessionData.SCAN_COUNT_TYPE=Dose","comparison_type":">=","value":"1"},]
        
    root_element = "xnat:subjectData"
    test = XMLCreator(root_element, search_field, search_where)

    
    params = {'format': 'csv'}
    
    files = {'file': open('example.xml', 'rb')}
    response = requests.post('http://128.16.11.124:8080/xnat/data/search/', params=params, files=files, auth=('admin', 'mortavar1977'))
    df = pd.read_csv(StringIO(response.text))
    print(df)
