#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HASHSIZE 17

struct member_list *hashtable[HASHSIZE];

struct member_list{
  char name[16];
  int score;
  struct member_list *next;
};


int hash(char *key){
  int hashval = 0;
  while(*key!=0){
    hashval += *key;
    key++;
  }

  return hashval % HASHSIZE;
}

void add(char *c, int score){
  int hashval;
  struct member_list *p, *ptr;
  hashval = hash(c);

  ptr = malloc(sizeof(struct member_list));
  strcpy(ptr->name,c);
  ptr->score = score;
  ptr->next=hashtable[hashval];
  hashtable[hashval]=ptr;
}

struct member_list *find(char *c){
  struct member_list *p;
  int hashval;


  hashval = hash(c);
  p = hashtable[hashval];

  while(p!=NULL){
    if(strcmp(c,p->name)==0){
      return p;
    }
    p = p->next;
  }
  return NULL;

}

int main(){
  char word[1024];
  int i;
  struct member_list *ptr;

  for(i=0;i<17;i++){
    hashtable[i] = NULL;
  }
  add("takimoto",42);
  add("katsurada",122);
  add("matsuzawa",35);
  add("ohmura",12);

  while(scanf("%s",word)!=EOF){
    ptr = find(word);
    if(ptr!=NULL){
      printf("%s, %i\n",ptr->name,ptr->score);
    }else{
      printf("none\n");
    }
  }
}
