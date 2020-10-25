// what u gonna do when wh offline, what u gonna do
#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(void)
{
    int ret;

		srand(time(NULL));
    HANDLE fh;
    WIN32_FIND_DATA fd;
		int c=0;
		int curc=0;
    while (true)
    {
			c=rand()%89;
			
			
        memset(&fd, 0x0, sizeof(fd));
        fh = FindFirstFile("e:\\work\\*.wav", &fd);
        // deb("dump fh: %x", fh);
				curc=c;
        if (fh != INVALID_HANDLE_VALUE)
        {
			      do
            {
							FindNextFile(fh, &fd);                
            }
            while (curc--);
						printf("%u", c);
        }
				
				char fn[MAX_PATH];
				sprintf(fn,"e:\\work\\%s",fd.cFileName);
        ret=PlaySound(fn,NULL,SND_FILENAME);
				curc=0;
       // printf("%d %d",ret,GetLastError());
    }
    return 0;

}

#just because not higher s
