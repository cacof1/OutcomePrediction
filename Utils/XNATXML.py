import xml.etree.cElementTree as ET
import requests
import pandas as pd
from io import StringIO

### Class to automatic create XML search from specified entry criteria and conditions using pycURL to be used for XNAT
### To explore available data, use curl -u user:pass -X GET  http://128.16.11.124:8080/xnat/data/search/elements/?format=csv

class XMLCreator():
    def __init__(self, root_element):#, search_field_list, search_where_list):

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
        
        self.root_element      = root_element
        ET.SubElement(self.bundle,"xdat:root_element_name").text = self.root_element ## Search uniquely against those         
        self.search_where_clause = ET.SubElement(self.bundle,"xdat:search_where", {"method":"AND"})
    def ConstructTree(self): return ET.tostring(self.bundle)

    def Add_search_field(self,search_field):
        root = ET.SubElement(self.bundle,"xdat:search_field")
        for key,value in search_field.items():
            ET.SubElement(root, "xdat:"+key).text = value

    def Add_search_where(self, search_where_list, search_where_method):
        child_set_clause = ET.SubElement(self.search_where_clause,"xdat:child_set", {"method":search_where_method})
        for search_where in search_where_list: ## All criterias
            criteria = ET.SubElement(child_set_clause, "xdat:criteria")
            for key,value in search_where.items():
                ET.SubElement(criteria, "xdat:"+key).text = value
    

