function cropped = cropper(panorama)
    cropped = panorama;

    for k = 1:2
        [height, width] = size(cropped);
        columncount = zeros(width);
        rowcount = zeros(height);
        % Count the number of black pixels in each row and column
        for j = 1:height
            for i = 1:width
                if panorama(j,i) == 0
                    columncount(i) = columncount(i) + 1;
                    rowcount(j) = rowcount(j) + 1;
                end
            end
        end
        
        j = 1;
        % skip rows with 1% or more black pixels
        while rowcount(j)/width >= 0.30+0.20*k && j<height
            j = j+1;
        end
        y1 = j;
        while rowcount(j)/width < 0.30+0.20*k && j<height
            j = j+1;
        end
        yh = j-y1;
        
        i = 1;
        % skip columns with 1% or more black pixels
        while columncount(i)/height >= 0.30+0.20*k && i<width
            i = i+1;
        end
        x1 = i;
        while columncount(i)/height < 0.30+0.20*k && i<width
            i = i+1;
        end
        xw = i-x1;
        
        cropped2 = imcrop(cropped, [x1  y1  xw  yh]);

        figure;
        imshow(cropped);
        hold on;
        plot([x1,x1],[y1,y1+yh],'Color','r','LineWidth',4, 'LineStyle','--');
        plot([x1,xw+x1],[y1,y1],'Color','r','LineWidth',4, 'LineStyle','--');
        plot([x1,xw+x1],[y1+yh,y1+yh],'Color','r','LineWidth',4, 'LineStyle','--');
        plot([xw+x1,xw+x1],[y1,y1+yh],'Color','r','LineWidth',4, 'LineStyle','--');
        hold off;

        cropped = cropped2;
    end
end