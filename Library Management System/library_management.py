# importing neccessary packages
from tkinter import *
from tkinter import ttk
import mysql.connector
from tkinter import messagebox
import tkinter
import datetime


class LibraryManagementSystem:
    def __init__(self,root):
        self.root=root
        self.root.title("Library Management System")
        self.root.geometry("1200x600+0+0")
        
        self.membertype_var=StringVar()
        self.prnno_var=StringVar()
        self.id_var=StringVar()
        self.firstname_var=StringVar()
        self.lastname_var=StringVar()
        self.address1_var=StringVar()
        self.address2_var=StringVar()
        self.postcode_var=StringVar()
        self.phoneno_var=StringVar()
        self.bookid_var=StringVar()
        self.booktitle_var=StringVar()
        self.author_var=StringVar()
        self.dateborrowed_var=StringVar()
        self.datedue_var=StringVar()
        self.daysonbook_var=StringVar()
        self.latereturnfine_var=StringVar()
        self.dateoverdue_var=StringVar()
        self.finalprice_var=StringVar()
        
        # MAIN TITLE
        lbltitle=Label(self.root,text="LIBRARY MANAGEMENT SYSTEM", bg="powder blue", fg="blue",bd=20,relief=RIDGE,font=("arial",25,"bold"),padx=2,pady=6)
        lbltitle.pack(side=TOP,fill=X)
        
        frame=Frame(self.root,bd=12,relief=RIDGE,padx=20,bg="powder blue")
        frame.place(x=0,y=90,width=1280,height=350)
        
        # DATAFRAME LEFT WITH LABELS
        DataFrameLeft=LabelFrame(frame,text="Library Membership Information", bg="powder blue", fg="green",bd=8,relief=RIDGE,font=("arial",10,"bold"))
        DataFrameLeft.place(x=0,y=5,width=600,height=310)
        
        lblMember=Label(DataFrameLeft,bg="powder blue",text="Member Type",font=("arial",8,"bold"),padx=2,pady=6)
        lblMember.grid(row=0,column=0,sticky=W)  #textvariable=self.member_var,
        
        comMember=ttk.Combobox(DataFrameLeft,textvariable=self.membertype_var,font=("arial",8,"bold"),width=24,state="readonly")
        comMember["value"]=("Student","Lecturer","Admin Staff",)
        comMember.current(0)
        comMember.grid(row=0,column=1)
        
        lblPRN_No=Label(DataFrameLeft,bg="powder blue",text="PRN No",font=("arial",8,"bold"),padx=2,pady=6)
        lblPRN_No.grid(row=1,column=0,sticky=W)
        txtPRN_No=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable=self.prnno_var,width=28)
        txtPRN_No.grid(row=1,column=1)
        
        lblID_NO=Label(DataFrameLeft,bg="powder blue",text="ID No",font=("arial",8,"bold"),padx=2,pady=6)
        lblID_NO.grid(row=2,column=0,sticky=W)
        txtID_NO=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable=self.id_var,width=28)
        txtID_NO.grid(row=2,column=1)
        
        lblFirstName=Label(DataFrameLeft,bg="powder blue",text="FirstName",font=("arial",8,"bold"),padx=2,pady=6)
        lblFirstName.grid(row=3,column=0,sticky=W)
        txtFirstName=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable= self.firstname_var,width=28)
        txtFirstName.grid(row=3,column=1)
        
        lblLastName=Label(DataFrameLeft,bg="powder blue",text="LastName",font=("arial",8,"bold"),padx=2,pady=6)
        lblLastName.grid(row=4,column=0,sticky=W)
        txtLastName=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable=self.lastname_var,width=28)
        txtLastName.grid(row=4,column=1)
        
        lblAddress1=Label(DataFrameLeft,bg="powder blue",text="Address1",font=("arial",8,"bold"),padx=2,pady=6)
        lblAddress1.grid(row=5,column=0,sticky=W)
        txtAddress1=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable=self.address1_var,width=28)
        txtAddress1.grid(row=5,column=1)
        
        lblAddress2=Label(DataFrameLeft,bg="powder blue",text="Address2",font=("arial",8,"bold"),padx=2,pady=6)
        lblAddress2.grid(row=6,column=0,sticky=W)
        txtAddress2=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable=self.address2_var,width=28)
        txtAddress2.grid(row=6,column=1)
        
        lblPostcode=Label(DataFrameLeft,bg="powder blue",text="Postcode",font=("arial",8,"bold"),padx=2,pady=6)
        lblPostcode.grid(row=7,column=0,sticky=W)
        txtPostcode=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable=self.postcode_var,width=28)
        txtPostcode.grid(row=7,column=1)
        
        lblPhone_No=Label(DataFrameLeft,bg="powder blue",text="Phone No",font=("arial",8,"bold"),padx=2,pady=6)
        lblPhone_No.grid(row=8,column=0,sticky=W)
        txtPhone_No=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable=self.phoneno_var,width=28)
        txtPhone_No.grid(row=8,column=1)
        
        lblBookId=Label(DataFrameLeft,bg="powder blue",text="Book Id",font=("arial",8,"bold"),padx=2,pady=6)
        lblBookId.grid(row=0,column=2,sticky=W)
        txtBookId=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable=self.bookid_var,width=28)
        txtBookId.grid(row=0,column=3)
        
        lblBookTitle=Label(DataFrameLeft,bg="powder blue",text="Book Title",font=("arial",8,"bold"),padx=2,pady=6)
        lblBookTitle.grid(row=1,column=2,sticky=W)
        txtBookTitle=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable=self.booktitle_var,width=28)
        txtBookTitle.grid(row=1,column=3)
        
        lblAuthor=Label(DataFrameLeft,bg="powder blue",text="Author",font=("arial",8,"bold"),padx=2,pady=6)
        lblAuthor.grid(row=2,column=2,sticky=W)
        txtAuthor=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable=self.author_var,width=28)
        txtAuthor.grid(row=2,column=3)
        
        lblDateBorrowed=Label(DataFrameLeft,bg="powder blue",text="Date Borrowed",font=("arial",8,"bold"),padx=2,pady=6)
        lblDateBorrowed.grid(row=3,column=2,sticky=W)
        txtDateBorrowed=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable=self.dateborrowed_var,width=28)
        txtDateBorrowed.grid(row=3,column=3)
        
        lblDateDue=Label(DataFrameLeft,bg="powder blue",text="Date Due",font=("arial",8,"bold"),padx=2,pady=6)
        lblDateDue.grid(row=4,column=2,sticky=W)
        txtDateDue=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable=self.datedue_var,width=28)
        txtDateDue.grid(row=4,column=3)
        
        lblDaysOnBook=Label(DataFrameLeft,bg="powder blue",text="Days On Book",font=("arial",8,"bold"),padx=2,pady=6)
        lblDaysOnBook.grid(row=5,column=2,sticky=W)
        txtDaysOnBook=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable=self.daysonbook_var,width=28)
        txtDaysOnBook.grid(row=5,column=3)
        
        lblLateReturnFine=Label(DataFrameLeft,bg="powder blue",text="Late Return Fine",font=("arial",8,"bold"),padx=2,pady=6)
        lblLateReturnFine.grid(row=6,column=2,sticky=W)
        txtLateReturnFine=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable=self.latereturnfine_var,width=28)
        txtLateReturnFine.grid(row=6,column=3)
        
        lblDateOverDate=Label(DataFrameLeft,bg="powder blue",text="Date Over Date",font=("arial",8,"bold"),padx=2,pady=6)
        lblDateOverDate.grid(row=7,column=2,sticky=W)
        txtDateOverDate=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable= self.dateoverdue_var,width=28)
        txtDateOverDate.grid(row=7,column=3)
        
        lblActualPrice=Label(DataFrameLeft,bg="powder blue",text="Actual Price",font=("arial",8,"bold"),padx=2,pady=6)
        lblActualPrice.grid(row=8,column=2,sticky=W)
        txtActualPrice=Entry(DataFrameLeft,font=("arial",8,"bold"),textvariable=self.finalprice_var,width=28)
        txtActualPrice.grid(row=8,column=3)
        
        # DATAFRAME RIGHT
        DataFrameRight=LabelFrame(frame,text="Book Details", bg="powder blue", fg="green",bd=8,relief=RIDGE,font=("arial",10,"bold"))
        DataFrameRight.place(x=610,y=5,width=605,height=310)
        
        self.txtBox=Text(DataFrameRight, font=("arial",10,"bold"),width=38,height=16,padx=2,pady=6)
        self.txtBox.grid(row=0,column=2)
        
        listScrollbar=Scrollbar(DataFrameRight)
        listScrollbar.grid(row=0,column=1,sticky="ns")
        
        listBooks=['Hamlet','Anna Karenina','A Passage to India','The Tempest','Romeo and Juliet','To Kill A Mocking Bird','Macbeth','The great Gatsby','Othello','Julius Caesar',
                 'Henry V','One Hundred Years of Solitude','King Lear','The Merchant of Venice',
                 'Invisible Man','Much Ado About Nothing','Beloved','Twelfth Night',"A Midsummer Night's Dream",]
        
        def SelectBook(event=""):
            value=str(listBox.get(listBox.curselection()))
            x=value
            if (x=="Hamlet"):
                self.bookid_var.set("Book1")
                self.booktitle_var.set("Fiction")
                self.author_var.set("William Shakespere")
                
                d1=datetime.datetime.today()
                d2=datetime.timedelta(days=15)
                d3=d1+d2
                self.dateborrowed_var.set(d1)
                self.datedue_var.set(d3)
                self.daysonbook_var.set(15)
                self.latereturnfine_var.set("Rs.50")
                self.dateoverdue_var.set("No")
                self.finalprice_var.set("Rs.500")
                
            elif (x=="Anna Karenina"):
                self.bookid_var.set("Book2")
                self.booktitle_var.set("Novel")
                self.author_var.set("Leo Tolstoy")
                
                d1=datetime.datetime.today()
                d2=datetime.timedelta(days=15)
                d3=d1+d2
                self.dateborrowed_var.set(d1)
                self.datedue_var.set(d3)
                self.daysonbook_var.set(15)
                self.latereturnfine_var.set("Rs.50")
                self.dateoverdue_var.set("No")
                self.finalprice_var.set("Rs.800")
                
            elif (x=="A Passage to India"):
                self.bookid_var.set("Book3")
                self.booktitle_var.set("History")
                self.author_var.set("E.M.Forster")
                
                d1=datetime.datetime.today()
                d2=datetime.timedelta(days=15)
                d3=d1+d2
                self.dateborrowed_var.set(d1)
                self.datedue_var.set(d3)
                self.daysonbook_var.set(15)
                self.latereturnfine_var.set("Rs.50")
                self.dateoverdue_var.set("No")
                self.finalprice_var.set("Rs.600")  
                # for future use
                     
        #To get books in listbox
        listBox=Listbox(DataFrameRight,font=("arial",10,"bold"),width=38,height=16)
        listBox.bind("<<ListboxSelect>>",SelectBook)
        listBox.grid(row=0,column=0,padx=4)
        listScrollbar.config(command=listBox.yview)
        
        for item in listBooks:
            listBox.insert(END,item)
        
        # BUTTONS FRAME
        Framebutton=Frame(self.root,bd=12,relief=RIDGE,padx=20,bg="powder blue")
        Framebutton.place(x=0,y=400,width=1280,height=60)
        
        btnAddData=Button(Framebutton,command=self.Add_data,text="Add Data",font=("arial",10,"bold"),width=23,bg="blue",fg="white")
        btnAddData.grid(row=0,column=0)
        
        btnShowData=Button(Framebutton,command=self.Show_Data,text="Show Data",font=("arial",10,"bold"),width=25,bg="blue",fg="white")
        btnShowData.grid(row=0,column=1)
        
        btnUpdate=Button(Framebutton,command=self.iUpdate,text="Update",font=("arial",10,"bold"),width=25,bg="blue",fg="white")
        btnUpdate.grid(row=0,column=2)
        
        btnDelete=Button(Framebutton,command=self.iDelete,text="Delete",font=("arial",10,"bold"),width=25,bg="blue",fg="white")
        btnDelete.grid(row=0,column=3)
        
        btnReset=Button(Framebutton,command=self.Reset,text="Reset",font=("arial",10,"bold"),width=25,bg="blue",fg="white")
        btnReset.grid(row=0,column=4)
        
        btnExit=Button(Framebutton,command=self.iExit,text="Exit",font=("arial",10,"bold"),width=23,bg="blue",fg="white")
        btnExit.grid(row=0,column=5)
        
        
         #INFORMATION FRAME
        FrameDetails=Frame(self.root,bd=12,relief=RIDGE,padx=20,bg="powder blue")
        FrameDetails.place(x=0,y=450,width=1280,height=200)
        
        Table_frame=Frame(FrameDetails,bd=6,relief=RIDGE,bg="powder blue")
        Table_frame.place(x=0,y=2,width=1250,height=180)
        
        xscroll=ttk.Scrollbar(Table_frame,orient=HORIZONTAL)
        yscroll=ttk.Scrollbar(Table_frame,orient=VERTICAL)
        
        self.library_table=ttk.Treeview(Table_frame,column=("membertype","prnno","idno","firstname","lastname",
                                        "address1","address2","postcode","phoneno","bookid","booktitle","author","dateborrowed",
                                        "datedue","days","latereturnfine","dateoverdue","finalprice"),xscrollcommand=xscroll.set,yscrollcommand=yscroll.set)
        
        xscroll.pack(side=BOTTOM,fill=X)
        yscroll.pack(side=RIGHT,fill=Y)
        
        xscroll.config(command=self.library_table.xview)
        yscroll.config(command=self.library_table.yview)
        
        self.library_table.heading("membertype",text="Member Type")
        self.library_table.heading("prnno",text="PRN No")
        self.library_table.heading("idno",text="ID No")
        self.library_table.heading("firstname",text="First Name")
        self.library_table.heading("lastname",text="Last Name")
        self.library_table.heading("address1",text="Address1")
        self.library_table.heading("address2",text="Address2")
        self.library_table.heading("postcode",text="Post Code")
        self.library_table.heading("phoneno",text="Phone No")
        self.library_table.heading("bookid",text="Book ID")
        self.library_table.heading("booktitle",text="Book Title")
        self.library_table.heading("author",text="Author")
        self.library_table.heading("dateborrowed",text="Date Borrowed")
        self.library_table.heading("datedue",text="Date Due")
        self.library_table.heading("days",text="Days on Book")
        self.library_table.heading("latereturnfine",text="Late Return Fine")
        self.library_table.heading("dateoverdue",text="Date Over Due")
        self.library_table.heading("finalprice",text="Final Price")
        
        self.library_table["show"]="headings"
        self.library_table.pack(fill=BOTH,expand=1)
        
        self.library_table.column("membertype",width=100)
        self.library_table.column("prnno",width=100)
        self.library_table.column("idno",width=100)
        self.library_table.column("firstname",width=100)
        self.library_table.column("lastname",width=100)
        self.library_table.column("address1",width=100)
        self.library_table.column("address2",width=100)
        self.library_table.column("postcode",width=100)
        self.library_table.column("phoneno",width=100)
        self.library_table.column("bookid",width=100)
        self.library_table.column("booktitle",width=100)
        self.library_table.column("author",width=100)
        self.library_table.column("dateborrowed",width=100)
        self.library_table.column("datedue",width=100)
        self.library_table.column("days",width=100)
        self.library_table.column("latereturnfine",width=100)
        self.library_table.column("dateoverdue",width=100)
        self.library_table.column("finalprice",width=100)
        
        self.fetch_data()
        self.library_table.bind("<ButtonRelease-1>",self.get_cursor)
        
        
    def Add_data(self):
        conn=mysql.connector.connect(host="localhost",username="root",password="Geethu@95",database="my_data")
        my_cursor=conn.cursor()
        my_cursor.execute("insert into library values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(
                                                                                                                self.membertype_var.get(),
                                                                                                                self.prnno_var.get(),
                                                                                                                self.id_var.get(),
                                                                                                                self.firstname_var.get(),
                                                                                                                self.lastname_var.get(),
                                                                                                                self.address1_var.get(),
                                                                                                                self.address2_var.get(),
                                                                                                                self.postcode_var.get(),
                                                                                                                self.phoneno_var.get(),
                                                                                                                self.bookid_var.get(),
                                                                                                                self.booktitle_var.get(),
                                                                                                                self.author_var.get(),
                                                                                                                self.dateborrowed_var.get(),
                                                                                                                self.datedue_var.get(),
                                                                                                                self.daysonbook_var.get(),
                                                                                                                self.latereturnfine_var.get(),
                                                                                                                self.dateoverdue_var.get(),
                                                                                                                self.finalprice_var.get()               
                                                                                                             ))
        conn.commit()
        self.fetch_data()
        conn.close()
        
        messagebox.showinfo("Success","Member has been inserted successfully")
     
     
    def iUpdate(self):
        conn=mysql.connector.connect(host="localhost",username="root",password="Geethu@95",database="my_data")
        my_cursor=conn.cursor()
        my_cursor.execute("update library set MemberType=%s,ID_No=%s,FirstName=%s,LastName=%s,Address1=%s,Address2=%s,PostCode=%s,PhoneNo=%s,BookId=%s,BookTitle=%s,Author=%s,DateBorrowed=%s,DateDue=%s,DaysofBook=%s,LateReturnFine=%s,DateOverDue=%s,FinalPrice=%s where PRN_No=%s",(      self.membertype_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.id_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.firstname_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.lastname_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.address1_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.address2_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.postcode_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.phoneno_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.bookid_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.booktitle_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.author_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.dateborrowed_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.datedue_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.daysonbook_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.latereturnfine_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.dateoverdue_var.get(),
                                                                                                                                                                                                                                                                                                                                                self.finalprice_var.get(), 
                                                                                                                                                                                                                                                                                                                                                self.prnno_var.get(), 
                                                                                                                                                                                                                                                                                                                                            ))
        conn.commit()
        self.fetch_data()
        # self.reset()
        conn.close()
        
        messagebox.showinfo("Success", "Member has been updated")
        
     
     
    def fetch_data(self):
        conn=mysql.connector.connect(host="localhost",username="root",password="Geethu@95",database="my_data")
        my_cursor=conn.cursor()
        my_cursor.execute("select * from library")
        rows=my_cursor.fetchall()
        
        if len(rows)!=0:
            self.library_table.delete(*self.library_table.get_children())
            for i in rows:
                self.library_table.insert("",END,values=i)
            conn.commit()
        conn.close()
        
        
    def get_cursor(self,event=""):
        cursor_row=self.library_table.focus()
        content=self.library_table.item(cursor_row)
        row=content['values']
        
        self.membertype_var.set(row[0])
        self.prnno_var.set(row[1])
        self.id_var.set(row[2])
        self.firstname_var.set(row[3])
        self.lastname_var.set(row[4])
        self.address1_var.set(row[5])
        self.address2_var.set(row[6])
        self.postcode_var.set(row[7])
        self.phoneno_var.set(row[8])
        self.bookid_var.set(row[9])
        self.booktitle_var.set(row[10])
        self.author_var.set(row[11])
        self.dateborrowed_var.set(row[12])
        self.datedue_var.set(row[13])
        self.daysonbook_var.set(row[14])
        self.latereturnfine_var.set(row[15])
        self.dateoverdue_var.set(row[16])
        self.finalprice_var.set(row[17])
        
    def Show_Data(self):          
        self.txtBox.insert(END,"Member Type\t\t" + self.membertype_var.get() + "\n")
        self.txtBox.insert(END,"PRN No\t\t" + self.prnno_var.get() + "\n")
        self.txtBox.insert(END,"ID No\t\t" + self.id_var.get() + "\n")
        self.txtBox.insert(END,"First Name\t\t" + self.firstname_var.get() + "\n")
        self.txtBox.insert(END,"Last Name\t\t" + self.lastname_var.get() + "\n")
        self.txtBox.insert(END,"Address1\t\t" + self.address1_var.get() + "\n")
        self.txtBox.insert(END,"Address2\t\t" + self.address2_var.get() + "\n")
        self.txtBox.insert(END,"Post Code\t\t" + self.postcode_var.get() + "\n")
        self.txtBox.insert(END,"Phone No\t\t" + self.phoneno_var.get() + "\n")
        self.txtBox.insert(END,"Book Id\t\t" + self.bookid_var.get() + "\n")
        self.txtBox.insert(END,"Book Title\t\t" + self.booktitle_var.get() + "\n")
        self.txtBox.insert(END,"Author\t\t" + self.author_var.get() + "\n")
        self.txtBox.insert(END,"Date Borrowed\t\t" + self.dateborrowed_var.get() + "\n")
        self.txtBox.insert(END,"Date Due\t\t" + self.datedue_var.get() + "\n")
        self.txtBox.insert(END,"Days on Book\t\t" + self.daysonbook_var.get() + "\n")
        self.txtBox.insert(END,"Late Return Fine\t\t" + self.latereturnfine_var.get() + "\n")
        self.txtBox.insert(END,"Date Over Due\t\t" + self.dateoverdue_var.get() + "\n")
        self.txtBox.insert(END,"Final Price\t\t" + self.finalprice_var.get() + "\n")
     
    def Reset(self):
        self.membertype_var.set(""),    
        self.prnno_var.set(""),
        self.id_var.set(""),
        self.firstname_var.set(""),
        self.lastname_var.set(""),
        self.address1_var.set(""),
        self.address2_var.set(""),
        self.postcode_var.set(""),
        self.phoneno_var.set(""),
        self.bookid_var.set(""),
        self.booktitle_var.set(""),
        self.author_var.set(""),
        self.dateborrowed_var.set(""),
        self.datedue_var.set(""),
        self.daysonbook_var.set(""),
        self.latereturnfine_var.set(""),
        self.dateoverdue_var.set(""),
        self.finalprice_var.set("")
        self.txtBox.delete("1.0",END)
        
    def iExit(self):
        iExit=tkinter.messagebox.askyesno("Library management System","Do you want to exit?")
        if iExit>0:
            self.root.destroy()
            return
        
        
    def iDelete(self):
        if self.prnno_var.get()=="" or self.id_var.get()=="":
            messagebox.showerror("Error","First Select the Member")
        else:
            conn=mysql.connector.connect(host="localhost",username="root",password="Geethu@95",database="my_data")
            my_cursor=conn.cursor()
            query="delete from library where PRN_No=%s"
            value=(self.prnno_var.get(),)
            my_cursor.execute(query,value)
            
            conn.commit()
            self.fetch_data()
            self.Reset()
            # self.close()
            
            messagebox.showinfo("Success","Member has been deleted")
            
    
        
if __name__ =="__main__":
    root=Tk()
    obj=LibraryManagementSystem(root)
    root.mainloop()  
      
         
        